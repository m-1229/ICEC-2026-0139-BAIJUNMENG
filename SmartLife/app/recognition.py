"""监听会话服务：状态机 + 工作线程 + 模拟音源 / 真实麦克风 + 内存事件环。

两条输入通路
------------
* `source="simulated"`：服务端固定映射的 48 kHz WAV，按 20 ms 节拍喂给
  `StreamingRecognizer.push()`（阶段 2 已验收的行为，本阶段不得回归）。
* `source="microphone"`：`sounddevice.InputStream` 打开真实输入流
  （48000 Hz / 单声道 / float32），回调只做 `copy()` + 入有界队列，
  推理全部在 worker 线程里做，**绝不在音频回调里跑 TensorFlow**。

硬边界（每一条都在代码里可审计）
------------------------------
1. **YAMNet 事件不写数据库**：`persist` 固定 False 且构造时断言；YAMNet
   触发只进内存 deque。本阶段**唯一**的落库通路是敲门声学候选桥接，
   见第 6 条，与 `persist` 相互独立。
2. **不存音频**：整条链路没有任何音频落盘调用，采集数据用完即弃；
3. **不伪造事件**：YAMNet 的正式事件仍来自同一套 `FrameJudge`（阈值 0.45、
   3 帧均值）；声学候选只能写成 `source='acoustic'`、`confidence=NULL` 的
   敲门事件，且只在真实麦克风会话内、每个簇只写一次。
4. **识别栈懒加载**：模块导入时不碰 numpy / TensorFlow，只在 worker 线程里装栈，
   因此主 venv（没有 numpy）也能正常 `import app.main`。
5. **麦克风开关**：`ALLOW_MICROPHONE`（默认开，可用环境变量
   `SMARTLIFE_ENABLE_MICROPHONE=0` 关闭）；关闭时 microphone 请求一律 400，
   自动化测试因此可以完全绕开音频硬件。
6. **敲门事件桥接**：`KNOCK_PERSIST_ENABLED`（**默认关**，需显式设置
   `SMARTLIFE_ENABLE_KNOCK_PERSIST=1` 才开启；`=0` 与未设置都关闭）。开启后，
   只有 `source="microphone"` 的会话、且 `candidate_type="knock"` 的候选，才会在
   每个簇**首次出现**时写一条 `event_type='knock' / source='acoustic' /
   confidence=NULL` 的正式事件；同一簇的后续更新只刷新内存里的候选视图。
   事件时间是 `listening_started_at + start_time_s`（首击起点），不是写库时刻。
   去重键是 `(session_id, detector.reset_count, cluster_id)` —— 单独用
   `cluster_id` 会在监听重启或检测器 reset 后把新簇误认成旧簇。
   写库失败只累计错误计数并把错误暴露到 `status.acoustic`，绝不冒泡到
   worker 线程；失败的簇不会被记为"已写"，下次更新会重试。


状态机
------
    idle --start--> loading --装栈/开流成功--> listening --stop/素材结束/异常--> stopping --> idle
                        |                                                        |
                        +--装栈失败 / 开流失败 / 设备断开-------------------> error
    error --start--> loading      （错误已记录，允许重试）
    loading/listening/stopping 期间 start -> 409 冲突

线程约定
--------
* 只有一个 worker 线程；状态、计数、事件受 `_lock` 保护，麦克风计数另有
  `session.mic.lock`（音频回调只碰这一把，避免和 API 线程抢同一把锁）。
* **join 一律在锁外**：worker 收尾也要拿锁，锁内 join 会死锁。
* `stop()` 超时（5 s）绝不谎报 idle：置 `error` 并保留线程引用，
  在该线程真正退出前 `start()` 会继续拒绝。

首帧时序（方案甲，warmup=2，沿用阶段 1 结论）
    首帧 0.975 s / 预热结束 1.455 s / 首次完成 3 帧判定 2.895 s
    `stop` -> `start` 重新计预热，与"只丢一次、不在分块或清理时重复"的结论一致。
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import queue
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .audio_devices import enumerate_input_devices
from .listen_schemas import (
    AcousticCandidate as AcousticCandidateView,
    AcousticStats,
    DeviceInfo,
    FrameScore,
    FrameStats,
    ListenEvent,
    ListenStartRequest,
    ListenStatus,
    MicStats,
    SessionDevice,
)

logger = logging.getLogger("smartlife.recognition")

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE_DIR = PROJECT_ROOT / "yamnet_prototype"
MODEL_SUBDIR = Path("models") / "yamnet"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


#: 麦克风采集开关。阶段 3 经批准打开；`SMARTLIFE_ENABLE_MICROPHONE=0` 可关闭，
#: 自动化测试用它保证"完全不触碰音频硬件"。
ALLOW_MICROPHONE = _env_flag("SMARTLIFE_ENABLE_MICROPHONE", True)

#: 阶段 3 仍然固定不落库；阶段 4 迁移完成后才允许改。
PERSIST_ENABLED = False

#: 敲门声学候选 → 正式事件的桥接开关。**默认关闭**，只有显式设置
#: `SMARTLIFE_ENABLE_KNOCK_PERSIST=1` 才允许写正式事件（自动化测试因此天然隔离
#: 数据库）。它与 `PERSIST_ENABLED` 是两件事：后者指 YAMNet 触发落库，仍未启用。
KNOCK_PERSIST_ENABLED = _env_flag("SMARTLIFE_ENABLE_KNOCK_PERSIST", False)

#: stop 等待 worker 退出的上限（秒）。超时即 error，不谎报 idle。
STOP_JOIN_TIMEOUT = 5.0
#: 进程关闭时等待 worker 的时间，短一些避免拖慢重启。
SHUTDOWN_JOIN_TIMEOUT = 3.0

#: 模拟音源每次喂入的样本数：960 / 48000 = 20 ms
BLOCK_SAMPLES = 960
SIMULATED_SAMPLE_RATE = 48000

# ---- 麦克风采集 -----------------------------------------------------------
#: 第一版只支持 48000 Hz / 单声道 / float32；不支持就报错，不降级到 44.1 kHz。
MIC_SAMPLE_RATE = 48000
MIC_CHANNELS = 1
MIC_DTYPE = "float32"
#: 每次回调的帧数：960 / 48000 = 20 ms，与模拟通路同节拍。
MIC_BLOCK_SAMPLES = 960
#: 有界队列上限：50 块 ≈ 1.0 s。溢出即丢块并触发重同步，绝不无限增长。
QUEUE_MAX_BLOCKS = 50
#: 队列取块的等待时长，同时充当"设备是否还活着"的巡检周期。
QUEUE_GET_TIMEOUT = 0.2
#: YAMNet 输入率（由 StreamResampler48k 3:1 抽取得到），只用于把丢块时间
#: 换算回 16 kHz 时间轴，不是可调参数。
MIC_TARGET_RATE = 16000

# ---- 诊断（只增加可观测性，不参与任何判定） -------------------------------
#: 电平窗口：50 块 × 20 ms = 1.0 s
LEVEL_WINDOW_BLOCKS = 50
#: 逐帧分数环形缓冲长度（2 帧/秒 → 保留约 20 s）
FRAME_SCORE_KEEP = 40
#: dBFS 下限：0 与数字零统一钳到这里，避免 -inf 无法序列化
DBFS_FLOOR = -160.0

#: 客户端只能传 clip 名称；真实路径由服务端固定映射（防止路径穿越）
CLIP_FILES: Dict[str, tuple[str, str]] = {
    "doorbell": ("门铃", "doorbell_classic(1).wav"),
    "knock": ("敲门", "knock_medium.wav"),
    "control": ("烟雾报警", "smoke_alarm_standard.wav"),
}

#: 事件最多保留的条数（内存环形缓冲）
MAX_EVENTS = 50
#: 影子候选只保留最近若干条，避免长时间监听无限占用内存。
ACOUSTIC_CANDIDATE_KEEP = 20

ACTIVE_STATES = ("loading", "listening", "stopping")


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class ListenError(RuntimeError):
    """监听服务的基类异常。"""


class ListenStateConflict(ListenError):
    """当前状态不允许该操作（HTTP 409）。"""

    def __init__(self, state: str, message: str):
        super().__init__(message)
        self.state = state


class MicrophoneDisabled(ListenError):
    """麦克风采集开关关闭（HTTP 400）。"""


class ClipNotSupported(ListenError):
    """素材名不在固定映射里，或素材采样率不是 48 kHz（HTTP 400）。"""


class DeviceNotAllowed(ListenError):
    """客户端指定的设备不在候选列表、被排除，或做不到 48k/单声道/float32（HTTP 400）。"""


class RecognitionUnavailable(ListenError):
    """识别栈不可用：缺 numpy / 缺模型 / 缺依赖（HTTP 503）。"""


class MicrophoneUnavailable(ListenError):
    """音频后端不可用、没有可用设备、开流失败（HTTP 503）。"""


class AudioDeviceLost(MicrophoneUnavailable):
    """运行中输入流停止（拔出、被独占、驱动异常）。"""


# --------------------------------------------------------------------------- #
# 能力探测（便宜、可随时调用；不加载 TensorFlow、不打开设备）
# --------------------------------------------------------------------------- #
def sounddevice_available() -> bool:
    """当前解释器里是否装了 sounddevice（只查 spec，不 import、不碰硬件）。"""
    try:
        return importlib.util.find_spec("sounddevice") is not None
    except Exception:  # pragma: no cover - 取决于运行环境
        return False


def microphone_capture_available() -> bool:
    """麦克风采集是否可能可用：开关打开 + 后端已安装。不打开设备。"""
    return bool(ALLOW_MICROPHONE) and sounddevice_available()


def probe_capability() -> tuple[bool, Optional[str]]:
    """判断识别栈是否可用。只做目录 stat 与 numpy import，不加载模型。"""
    if not PROTOTYPE_DIR.is_dir():
        return False, f"找不到识别原型目录：{PROTOTYPE_DIR}"
    model_dir = PROTOTYPE_DIR / MODEL_SUBDIR
    if not (model_dir / "saved_model.pb").exists():
        return False, f"找不到模型文件：{model_dir / 'saved_model.pb'}"
    try:
        import numpy  # noqa: F401
    except Exception as exc:
        return False, (
            f"当前 Python 环境缺少 numpy，识别不可用（{type(exc).__name__}）。"
            f"请用 yamnet_prototype 的虚拟环境启动。"
        )
    return True, None


# --------------------------------------------------------------------------- #
# 音频后端接入点（测试可替换这两个函数，从而完全不碰硬件）
# --------------------------------------------------------------------------- #
def _import_sounddevice():
    try:
        import sounddevice as sd
    except Exception as exc:
        raise MicrophoneUnavailable(
            f"音频后端不可用（无法导入 sounddevice）：{type(exc).__name__}: {exc}"
        ) from exc
    return sd


def _check_input_settings(
    device_index: int,
    sample_rate: int = MIC_SAMPLE_RATE,
    channels: int = MIC_CHANNELS,
    dtype: str = MIC_DTYPE,
) -> None:
    """开流前复核设备能力。不打开流、不录音，只查询驱动是否接受这组参数。"""
    sd = _import_sounddevice()
    try:
        sd.check_input_settings(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
        )
    except Exception as exc:
        raise DeviceNotAllowed(
            f"设备 {device_index} 不支持 {sample_rate} Hz / {channels} 声道 / {dtype}："
            f"{type(exc).__name__}: {exc}（不做重采样降级）"
        ) from exc


# --------------------------------------------------------------------------- #
# 诊断：电平与分数（纯观测，绝不参与判定，也绝不改变入队/推力顺序）
# --------------------------------------------------------------------------- #
def _dbfs(value: float) -> float:
    """线性幅度 -> dBFS；0 钳到 DBFS_FLOOR（不回 -inf，便于 JSON）。"""
    if value <= 0.0:
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 20.0 * math.log10(value))


def _block_peak_ss(block) -> tuple[float, float]:
    """返回 (峰值, 平方和)。

    纯 Python 实现：**不依赖 numpy**，所以主 venv（没有 numpy）也能 import 本模块；
    960 样本/块的成本约几十微秒，对 20 ms 的回调节拍可以忽略。
    """
    peak = 0.0
    ss = 0.0
    for value in block:
        v = float(value)
        a = -v if v < 0.0 else v
        if a > peak:
            peak = a
        ss += v * v
    return peak, ss


def _opt_float(value) -> Optional[float]:
    """把可能是 None / numpy 标量的分数转成 float；转换失败返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _update_level(mic, block) -> None:
    """把一块音频的实测电平并入 `_MicState`（在回调线程里调用）。

    任何异常都被吞掉：诊断绝不能让采集或入队失败。
    """
    try:
        peak, ss = _block_peak_ss(block)
    except Exception:
        return
    try:
        with mic.lock:
            mic.level_blocks += 1
            mic.level_peak = peak
            if peak > mic.level_peak_max:
                mic.level_peak_max = peak
            if peak == 0.0:
                mic.zero_blocks += 1
            elif peak < 1e-6:
                mic.silent_blocks += 1
            mic.recent_peaks.append(peak)
            mic.recent_levels.append((ss, len(block)))
    except Exception:  # pragma: no cover - 锁/容器异常时也不能影响采集
        return


def _com_apartment_enter() -> bool:
    """在**当前线程**初始化 COM 单元，返回 True 表示需要在退出时 CoUninitialize。

    为什么必须有这一步
    ------------------
    Windows 的 WASAPI 输入流是在 COM 单元里创建的：`import sounddevice` 时
    `Pa_Initialize()` 在主线程执行，COM 单元只属于主线程。worker 线程默认没有
    单元，直接 `InputStream.start()` 会得到
    `PortAudioError -9999 ... 'WdmSyncIoctl: DeviceIoControl GLE = 0x490'`。
    实测：同一设备在主线程能开、在未初始化 COM 的线程开不了；调用
    `CoInitializeEx(COINIT_APARTMENTTHREADED)` 后即可正常打开（与 TensorFlow 无关）。

    非 Windows 平台直接返回 False，不做任何事。
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        hresult = ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        logger.warning("CoInitializeEx 失败，继续尝试开流：%s", exc)
        return False
    # S_OK(0) / S_FALSE(1) 需要配对 CoUninitialize；RPC_E_CHANGED_MODE 不需要。
    return hresult in (0, 1)


def _com_apartment_leave(entered: bool) -> None:
    if not entered:
        return
    try:
        import ctypes

        ctypes.windll.ole32.CoUninitialize()
    except Exception:  # pragma: no cover
        pass


def _open_microphone(
    device_index: int,
    sample_rate: int,
    blocksize: int,
    callback,
    finished_callback,
):
    """打开并启动输入流（必须在已初始化 COM 单元的采集线程里调用）。"""
    sd = _import_sounddevice()
    try:
        stream = sd.InputStream(
            device=device_index,
            channels=MIC_CHANNELS,
            samplerate=sample_rate,
            dtype=MIC_DTYPE,
            blocksize=blocksize,
            callback=callback,
            finished_callback=finished_callback,
            latency="low",
        )
        stream.start()
    except Exception as exc:
        raise MicrophoneUnavailable(
            f"无法打开麦克风设备 {device_index}（{sample_rate} Hz）："
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return stream


# --------------------------------------------------------------------------- #
# 识别栈（只在 worker 线程里装配）
# --------------------------------------------------------------------------- #
@dataclass
class _Stack:
    """把懒加载得到的模块与常量打包，供会话使用。"""

    np: object
    bundle: object
    read_wav: object
    recognizer_cls: object
    acoustic_cls: object
    window: int
    hop: int
    warmup_frames: int
    theta: float
    smooth_n: int
    cooldown_s: int = 0

    def config(self) -> Dict[str, float | int | str]:
        return {
            "window_samples": self.window,
            "hop_samples": self.hop,
            "warmup_frames": self.warmup_frames,
            "theta": self.theta,
            "smooth_n": self.smooth_n,
            "cooldown_s": self.cooldown_s,
            "acoustic_shadow_only": 1,
        }

    def new_recognizer(self, input_rate: int = SIMULATED_SAMPLE_RATE):
        """新会话必须新建识别器，并显式 reset_session()（调用方负责调用）。"""
        return self.recognizer_cls(
            self.bundle,
            input_rate=int(input_rate),
            theta=self.theta,
            smooth_n=self.smooth_n,
            cooldown_s=self.cooldown_s,
            warmup_frames=self.warmup_frames,
        )

    def new_acoustic(self, input_rate: int = SIMULATED_SAMPLE_RATE):
        """新会话创建一个独立的声学影子检测器。"""
        return self.acoustic_cls(sample_rate=int(input_rate))

    def load_clip(self, clip: str) -> object:
        """读取固定素材，返回 48 kHz float32 单声道波形。"""
        entry = CLIP_FILES.get(clip)
        if entry is None:
            raise ClipNotSupported(f"不支持的模拟素材：{clip}")
        path = PROTOTYPE_DIR / "recordings" / entry[0] / entry[1]
        if not path.exists():
            raise RecognitionUnavailable(f"模拟素材缺失：{path}")

        data, info = self.read_wav(str(path))
        sample_rate = int(info["sample_rate"])
        if sample_rate != SIMULATED_SAMPLE_RATE:
            raise ClipNotSupported(
                f"素材 {entry[1]} 是 {sample_rate} Hz，阶段 2 只接受 "
                f"{SIMULATED_SAMPLE_RATE} Hz（不做重采样降级）。"
            )
        np = self.np
        mono = data.mean(axis=1) if info["channels"] > 1 else data[:, 0]
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        mono = np.clip(mono, -1.0, 1.0)
        return np.ascontiguousarray(mono, dtype=np.float32)


def _load_stack() -> _Stack:
    """懒加载识别栈。失败抛 RecognitionUnavailable / 原始异常由 worker 记录。"""
    import sys

    if str(PROTOTYPE_DIR) not in sys.path:
        sys.path.insert(0, str(PROTOTYPE_DIR))

    import numpy as np

    from yamnet_batch_eval import read_wav
    from yamnet_recognizer import load_model
    from acoustic_features import AcousticShadowDetector
    from yamnet_stream import (
        HOP,
        SMOOTH_N,
        THETA_DEFAULT,
        WARMUP_DROP_FRAMES,
        WINDOW,
        StreamingRecognizer,
    )

    bundle = load_model()
    return _Stack(
        np=np,
        bundle=bundle,
        read_wav=read_wav,
        recognizer_cls=StreamingRecognizer,
        acoustic_cls=AcousticShadowDetector,
        window=int(WINDOW),
        hop=int(HOP),
        warmup_frames=int(WARMUP_DROP_FRAMES),
        theta=float(THETA_DEFAULT),
        smooth_n=int(SMOOTH_N),
        cooldown_s=0,
    )


def _iter_blocks(wave, loop: bool, speed: float, stop_event: threading.Event) -> Iterator:
    """按 20 ms 节拍产出音频块；stop_event 可立即打断等待。"""
    total = int(wave.size)
    if total == 0:
        return
    pos = 0
    interval = (BLOCK_SAMPLES / float(SIMULATED_SAMPLE_RATE)) / max(speed, 0.1)
    deadline = time.monotonic()
    while not stop_event.is_set():
        block = wave[pos:pos + BLOCK_SAMPLES]
        if block.size == 0:
            if not loop:
                return
            pos = 0
            continue
        pos += block.size
        yield block
        deadline += interval
        delay = deadline - time.monotonic()
        if delay > 0:
            if stop_event.wait(delay):
                return
        elif -delay > 2.0:
            # 落后超过 2 s（例如被系统挂起），重置节拍，避免一次性追赶
            deadline = time.monotonic()


# --------------------------------------------------------------------------- #
# 敲门事件桥接（唯一落库通路）
# --------------------------------------------------------------------------- #
def _persist_knock_event(occurred_at: datetime):
    """写入一条敲门声学事件；延迟 import，避免模块导入期就拖入 sqlite。

    单独做成模块级函数（而不是内联调用）有两个目的：
    1. 测试可以把 `app.recognition._persist_knock_event` 换成一个记录器，
       在不碰数据库的前提下验证"每个簇只写一次"；
    2. 让"写库"这个副作用在代码里只有一个入口，便于审计。
    """
    from .database import create_acoustic_knock_event

    return create_acoustic_knock_event(occurred_at)


# --------------------------------------------------------------------------- #
# 会话内部状态
# --------------------------------------------------------------------------- #
@dataclass
class _MicState:
    """麦克风通路的私有计数。音频回调只碰这把锁。"""

    lock: threading.Lock = field(default_factory=threading.Lock)
    queue: Optional["queue.Queue"] = None
    queue_max_blocks: int = 0
    dropped_blocks: int = 0
    dropped_samples: int = 0
    overflow_events: int = 0
    resync_count: int = 0
    callback_errors: int = 0
    input_flags: int = 0
    pending_resync: bool = False
    fatal: Optional[str] = None
    finished: bool = False
    #: 因丢块而"欠下"的音频时长（秒）。事件 time_s 要加上它，时间轴才不会倒退。
    offset_s: float = 0.0
    # ---- 电平诊断（回调里实测，不参与判定） ----
    level_peak: float = 0.0
    level_peak_max: float = 0.0
    level_blocks: int = 0
    zero_blocks: int = 0
    silent_blocks: int = 0
    #: 最近 1 秒的逐块峰值与 (平方和, 样本数)，用于 1 s 窗口的 max / RMS
    recent_peaks: deque = field(
        default_factory=lambda: deque(maxlen=LEVEL_WINDOW_BLOCKS)
    )
    recent_levels: deque = field(
        default_factory=lambda: deque(maxlen=LEVEL_WINDOW_BLOCKS)
    )


@dataclass
class _Session:
    session_id: str
    source: str
    clip: Optional[str]
    loop: bool
    speed: float
    started_at: datetime
    started_mono: float
    device_index: Optional[int] = None
    device_name: str = "(未启动)"
    device_hostapi: str = "none"
    device_sample_rate: int = SIMULATED_SAMPLE_RATE
    channels: Optional[int] = None
    dtype: Optional[str] = None
    listening_started_mono: Optional[float] = None
    #: 与 `listening_started_mono` 同时记录的 UTC 墙钟，仅用于把候选的相对秒
    #: （`start_time_s`）还原成事件时间。不能用 `started_at` 代替：后者是会话
    #: 创建时刻，比真正开始采集早了一个装栈周期（数秒）。
    listening_started_at: Optional[datetime] = None
    recognizer: object = None
    acoustic: object = None
    stream: object = None
    opened: bool = False
    stop_reason: Optional[str] = None
    mic: _MicState = field(default_factory=_MicState)


# --------------------------------------------------------------------------- #
# 服务
# --------------------------------------------------------------------------- #
class ListenService:
    """监听会话的唯一入口。进程内单例，由 API 层与 lifespan 使用。"""

    def __init__(self, persist: bool = False,
                 knock_persist: bool = KNOCK_PERSIST_ENABLED):
        if persist:
            raise ValueError(
                "YAMNet 事件持久化仍未启用：本阶段只桥接敲门声学事件，"
                "请在 persist=False 下使用 knock_persist。"
            )
        self.persist = False
        self._lock = threading.RLock()
        self._stream_lock = threading.Lock()
        self._state: str = "idle"
        self._session: Optional[_Session] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._events: deque[ListenEvent] = deque(maxlen=MAX_EVENTS)
        self._seq = 0
        self._frames = FrameStats()
        self._stream_pos = 0
        self._buffered = 0
        self._triggers = 0
        self._last_error: Optional[str] = None
        self._config: Dict[str, float | int | str] = {}
        #: 逐帧分数环形缓冲（诊断用，不影响判定）
        self._frame_scores: deque = deque(maxlen=FRAME_SCORE_KEEP)
        #: 本会话各组 3 帧均值的峰值（诊断用）
        self._peak_smoothed: Dict[str, float] = {}
        #: 声学影子状态；与 YAMNet 判定完全隔离。
        self._acoustic_state: Dict[str, object] = {}
        self._acoustic_candidates: deque[AcousticCandidateView] = deque(
            maxlen=ACOUSTIC_CANDIDATE_KEEP
        )
        self._acoustic_session_resets = 0
        self._acoustic_resync_resets = 0
        self._acoustic_errors = 0
        self._acoustic_last_error: Optional[str] = None
        # ---- 敲门正式事件桥接 ----
        self._knock_persist = bool(knock_persist)
        #: 已写成功的簇去重键：(session_id, detector 重置代数, cluster_id)
        self._knock_persisted: set[tuple[str, int, str]] = set()
        #: 正在写库的簇键；防止同一簇并发重复插入。失败即撤销，成功才转入上集合。
        self._knock_inflight: set[tuple[str, int, str]] = set()
        self._knock_events_persisted = 0
        self._knock_persist_errors = 0
        self._knock_persist_last_error: Optional[str] = None
        self._knock_last_event_id: Optional[int] = None

    # -- 只读 ---------------------------------------------------------------
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def available(self) -> bool:
        """识别栈是否可用（给 /api/health 用）。"""
        return probe_capability()[0]

    def status(self) -> ListenStatus:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> ListenStatus:
        now = time.monotonic()
        session = self._session
        uptime = (now - session.started_mono) if session else 0.0
        listening = (
            (now - session.listening_started_mono)
            if session and session.listening_started_mono
            else 0.0
        )
        if session is None:
            device = SessionDevice(
                index=None, name="(未启动)", hostapi="none", sample_rate=0, opened=False
            )
            mic = MicStats()
            source = None
        else:
            device = SessionDevice(
                index=session.device_index,
                name=session.device_name,
                hostapi=session.device_hostapi,
                sample_rate=session.device_sample_rate,
                opened=session.opened,
                channels=session.channels,
                dtype=session.dtype,
            )
            source = session.source
            if session.source == "microphone":
                state = session.mic
                with state.lock:
                    recent_peaks = list(state.recent_peaks)
                    window_ss = sum(item[0] for item in state.recent_levels)
                    window_n = sum(item[1] for item in state.recent_levels)
                    mic = MicStats(
                        block_samples=MIC_BLOCK_SAMPLES,
                        queue_max_blocks=state.queue_max_blocks,
                        queue_depth=state.queue.qsize() if state.queue is not None else 0,
                        dropped_blocks=state.dropped_blocks,
                        overflow_events=state.overflow_events,
                        resync_count=state.resync_count,
                        callback_errors=state.callback_errors,
                        input_flags=state.input_flags,
                        timeline_offset_s=round(state.offset_s, 3),
                        level_peak=round(state.level_peak, 6),
                        level_peak_dbfs=round(_dbfs(state.level_peak), 1),
                        level_peak_recent=round(max(recent_peaks) if recent_peaks else 0.0, 6),
                        level_peak_max=round(state.level_peak_max, 6),
                        level_rms_dbfs=round(
                            _dbfs(math.sqrt(window_ss / window_n) if window_n else 0.0), 1
                        ),
                        level_blocks=state.level_blocks,
                        zero_blocks=state.zero_blocks,
                        silent_blocks=state.silent_blocks,
                    )
            else:
                mic = MicStats()

        acoustic = AcousticStats(
            shadow_only=True,
            frames_processed=int(self._acoustic_state.get("frames_processed", 0)),
            samples_processed=int(self._acoustic_state.get("samples_processed", 0)),
            noise_floor_dbfs=float(
                self._acoustic_state.get("noise_floor_dbfs", -160.0)
            ),
            calibrated=bool(self._acoustic_state.get("calibrated", False)),
            current_relative_db=self._acoustic_state.get("current_relative_db"),  # type: ignore[arg-type]
            current_spectral_flatness=self._acoustic_state.get("current_spectral_flatness"),  # type: ignore[arg-type]
            current_spectral_centroid_hz=self._acoustic_state.get("current_spectral_centroid_hz"),  # type: ignore[arg-type]
            current_spectral_similarity=self._acoustic_state.get("current_spectral_similarity"),  # type: ignore[arg-type]
            current_tonal=bool(self._acoustic_state.get("current_tonal", False)),
            pending_impacts=int(self._acoustic_state.get("pending_impacts", 0)),
            recent_impacts=int(self._acoustic_state.get("recent_impacts", 0)),
            last_reset_reason=str(
                self._acoustic_state.get("last_reset_reason", "session")
            ),
            reset_count=int(self._acoustic_state.get("reset_count", 0)),
            session_resets=self._acoustic_session_resets,
            resync_resets=self._acoustic_resync_resets,
            errors=self._acoustic_errors,
            last_error=self._acoustic_last_error,
            knock_persist_enabled=self._knock_persist,
            persisted_events=self._knock_events_persisted,
            persist_errors=self._knock_persist_errors,
            persist_last_error=self._knock_persist_last_error,
            last_event_id=self._knock_last_event_id,
            candidates=list(self._acoustic_candidates),
        )

        return ListenStatus(
            state=self._state,  # type: ignore[arg-type]
            source=source,
            clip=session.clip if session else None,
            persist=self.persist,
            microphone_enabled=ALLOW_MICROPHONE,
            session_id=session.session_id if session else None,
            started_at=session.started_at if session else None,
            uptime_s=round(max(uptime, 0.0), 3),
            listening_s=round(max(listening, 0.0), 3),
            device=device,
            frames=self._frames.model_copy(),
            mic=mic,
            frame_scores=list(self._frame_scores),
            peak_smoothed=dict(self._peak_smoothed),
            acoustic=acoustic,
            stream_pos=self._stream_pos,
            buffered=self._buffered,
            triggers=self._triggers,
            events=list(self._events),
            last_error=self._last_error,
            stop_reason=session.stop_reason if session else None,
            config=dict(self._config),
        )

    # -- 启动 ---------------------------------------------------------------
    def start(self, request: ListenStartRequest) -> ListenStatus:
        if request.source == "microphone":
            return self._start_microphone(request)
        return self._start_simulated(request)

    def _start_simulated(self, request: ListenStartRequest) -> ListenStatus:
        if request.clip not in CLIP_FILES:
            raise ClipNotSupported(f"不支持的模拟素材：{request.clip}")

        ok, reason = probe_capability()
        if not ok:
            raise RecognitionUnavailable(reason or "识别栈不可用")

        with self._lock:
            self._ensure_startable()
            session = _Session(
                session_id=uuid.uuid4().hex[:12],
                source="simulated",
                clip=request.clip,
                loop=bool(request.loop),
                speed=float(request.speed),
                started_at=datetime.now(timezone.utc),
                started_mono=time.monotonic(),
                device_index=None,
                device_name=f"simulated:{request.clip}",
                device_hostapi="none",
                device_sample_rate=SIMULATED_SAMPLE_RATE,
                channels=None,
                dtype=None,
            )
            self._begin_session(session)
            return self._snapshot()

    def _start_microphone(self, request: ListenStartRequest) -> ListenStatus:
        if not ALLOW_MICROPHONE:
            raise MicrophoneDisabled(
                "麦克风采集已关闭（SMARTLIFE_ENABLE_MICROPHONE=0 / "
                "ALLOW_MICROPHONE=False），当前只允许 source='simulated'。"
            )

        ok, reason = probe_capability()
        if not ok:
            raise RecognitionUnavailable(reason or "识别栈不可用")

        device = self._resolve_device(request.device)

        # 开流前复核：不通过就明确报错，绝不"假装启动成功"、绝不降级到 44.1 kHz。
        _check_input_settings(int(device.index))

        with self._lock:
            self._ensure_startable()
            session = _Session(
                session_id=uuid.uuid4().hex[:12],
                source="microphone",
                clip=None,
                loop=False,
                speed=1.0,
                started_at=datetime.now(timezone.utc),
                started_mono=time.monotonic(),
                device_index=int(device.index),
                device_name=device.name,
                device_hostapi=device.hostapi,
                device_sample_rate=MIC_SAMPLE_RATE,
                channels=MIC_CHANNELS,
                dtype=MIC_DTYPE,
            )
            self._begin_session(session)
            return self._snapshot()

    def _resolve_device(self, requested: Optional[int]) -> DeviceInfo:
        """在**当前**设备列表里解析设备；客户端索引必须命中且未被排除。"""
        listing = enumerate_input_devices()
        if not listing.available:
            raise MicrophoneUnavailable(
                listing.detail or "音频后端不可用，无法枚举输入设备。"
            )
        if requested is None:
            target = listing.target
            if target is None:
                raise MicrophoneUnavailable(
                    "没有可用的物理输入设备（列表为空或全部被排除为虚拟/回环设备）。"
                )
            return target
        for device in listing.devices:
            if device.index == int(requested):
                if device.excluded:
                    raise DeviceNotAllowed(
                        f"设备 {device.index}「{device.name}」已被排除："
                        f"{device.exclude_reason or '虚拟或回环设备'}"
                    )
                return device
        raise DeviceNotAllowed(
            f"设备索引 {requested} 不在当前输入设备列表中"
            f"（共 {len(listing.devices)} 个输入设备）。"
        )

    def _ensure_startable(self) -> None:
        """调用方必须已持锁。检查状态与残留线程，并重置会话级状态。"""
        if self._state in ACTIVE_STATES:
            raise ListenStateConflict(
                self._state, f"当前状态为 {self._state}，不能重复启动监听。"
            )
        if self._thread is not None and self._thread.is_alive():
            raise ListenStateConflict(
                "error", "上一个会话的工作线程仍在退出，请稍后重试。"
            )

    def _begin_session(self, session: _Session) -> None:
        """调用方必须已持锁。重置会话级状态并起 worker。"""
        self._stop_event = threading.Event()
        self._session = session
        self._events.clear()
        self._seq = 0
        self._triggers = 0
        self._frames = FrameStats()
        self._stream_pos = 0
        self._buffered = 0
        self._last_error = None
        self._config = {}          # 装栈成功后才填，避免新会话显示上一会话的配置
        self._frame_scores.clear()
        self._peak_smoothed = {}
        self._acoustic_state = {}
        self._acoustic_candidates.clear()
        self._acoustic_session_resets = 0
        self._acoustic_resync_resets = 0
        self._acoustic_errors = 0
        self._acoustic_last_error = None
        # 新会话的 cluster_id 会从 knock-1 重新计数，去重键里的 session_id 保证
        # 不会被当成上一会话的旧簇；这里顺带清空集合，避免长时间运行无限增长。
        self._knock_persisted.clear()
        self._knock_inflight.clear()
        self._knock_events_persisted = 0
        self._knock_persist_errors = 0
        self._knock_persist_last_error = None
        self._knock_last_event_id = None
        self._state = "loading"

        thread = threading.Thread(
            target=self._worker,
            args=(session, self._stop_event),
            name=f"listen-worker-{session.session_id}",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    # -- 停止 ---------------------------------------------------------------
    def stop(self) -> ListenStatus:
        with self._lock:
            state = self._state
            if state == "idle":
                return self._snapshot()
            if state == "error" and not self._thread_alive():
                self._stop_event.set()
                return self._snapshot()
            if state == "stopping":
                thread = self._thread
            else:
                self._state = "stopping"
                self._stop_event.set()
                thread = self._thread
            session = self._session

        # 先关流：设备马上释放，worker 也会立刻从 queue.get 里醒过来。
        if session is not None:
            self._close_stream(session)

        if thread is None:
            with self._lock:
                if self._state == "stopping":
                    self._state = "idle"
                return self._snapshot()

        # 锁外 join：worker 收尾要拿同一把锁
        thread.join(timeout=STOP_JOIN_TIMEOUT)

        with self._lock:
            if thread.is_alive():
                # 绝不谎报 idle：线程还活着就保持 error，且 start() 会继续拒绝
                self._state = "error"
                self._last_error = (
                    f"停止超时：工作线程在 {STOP_JOIN_TIMEOUT:g} 秒内未退出，"
                    f"监听未完全关闭。"
                )
                if self._session is not None and not self._session.stop_reason:
                    self._session.stop_reason = "stop_timeout"
            elif self._state == "stopping":
                self._state = "idle"
                if self._session is not None and not self._session.stop_reason:
                    self._session.stop_reason = "stopped"
            return self._snapshot()

    def shutdown(self) -> None:
        """进程关闭时调用；只在 lifespan 收尾处使用。"""
        self._stop_event.set()
        session = self._session
        if session is not None:
            self._close_stream(session)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT)
        with self._lock:
            if thread is not None and thread.is_alive():
                self._state = "error"
                self._last_error = "进程关闭时工作线程未在限定时间内退出。"

    # -- 音频流 -------------------------------------------------------------
    def _close_stream(self, session: _Session) -> None:
        """幂等关闭输入流（stop / 异常 / 进程退出都走这里）。"""
        with self._stream_lock:
            stream = session.stream
            session.stream = None
            session.opened = False
        if stream is None:
            return
        for action in ("stop", "close"):
            try:
                getattr(stream, action)()
            except Exception as exc:  # 已停止 / 已关闭都不算错误
                logger.warning("关闭音频流 %s() 失败：%s", action, exc)
        logger.info("麦克风输入流已关闭（device=%s）", session.device_name)

    @staticmethod
    def _stream_alive(session: _Session) -> bool:
        stream = session.stream
        if stream is None:
            return False
        try:
            return bool(getattr(stream, "active", True))
        except Exception:
            return False

    # -- 内部 ---------------------------------------------------------------
    def _thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _worker(self, session: _Session, stop_event: threading.Event) -> None:
        try:
            stack = _load_stack()
            if stop_event.is_set():
                session.stop_reason = "stopped_during_loading"
                return
            if session.source == "microphone":
                self._run_microphone(session, stop_event, stack)
            else:
                self._run_simulated(session, stop_event, stack)

        except AudioDeviceLost as exc:
            with self._lock:
                self._state = "error"
                self._last_error = str(exc)
                session.stop_reason = "device_lost"
        except Exception as exc:  # 装栈失败、素材缺失、开流失败、推理异常
            with self._lock:
                self._state = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                session.stop_reason = "error"
        finally:
            self._close_stream(session)
            with self._lock:
                if self._state != "error":
                    self._state = "idle"
                if self._thread is threading.current_thread():
                    self._thread = None

    # -- 通路 A：模拟素材（阶段 2 行为，保持不变） ---------------------------
    def _run_simulated(self, session: _Session, stop_event: threading.Event,
                       stack: _Stack) -> None:
        wave = stack.load_clip(session.clip)
        # 不传 input_rate：保持阶段 2 的调用签名，假识别栈（测试注入）也能直接替换
        recognizer = stack.new_recognizer()
        recognizer.reset_session()   # 要求：每次新会话显式重置
        acoustic = self._make_acoustic(stack, session)

        with self._lock:
            if stop_event.is_set():
                session.stop_reason = "stopped_during_loading"
                return
            session.recognizer = recognizer
            session.listening_started_mono = time.monotonic()
            session.listening_started_at = datetime.now(timezone.utc)
            self._config = stack.config()
            self._state = "listening"

        for block in _iter_blocks(wave, session.loop, session.speed, stop_event):
            self._shadow_push(acoustic, block, session)
            records = recognizer.push(block)
            self._sync(recognizer, records)

        if stop_event.is_set():
            session.stop_reason = "stopped"
        else:
            self._sync(recognizer, recognizer.flush())
            session.stop_reason = "clip_finished"

    # -- 通路 B：真实麦克风 -------------------------------------------------
    def _run_microphone(self, session: _Session, stop_event: threading.Event,
                        stack: _Stack) -> None:
        # WASAPI 需要一个属于本线程的 COM 单元，否则 start() 会报 WdmSyncIoctl。
        com_entered = _com_apartment_enter()
        try:
            self._microphone_session(session, stop_event, stack)
        finally:
            self._close_stream(session)
            _com_apartment_leave(com_entered)

    def _microphone_session(self, session: _Session, stop_event: threading.Event,
                            stack: _Stack) -> None:
        recognizer = stack.new_recognizer(input_rate=MIC_SAMPLE_RATE)
        recognizer.reset_session()   # 新会话重新执行方案甲的两帧预热
        acoustic = self._make_acoustic(stack, session)

        mic = session.mic
        audio_queue: "queue.Queue" = queue.Queue(maxsize=max(1, int(QUEUE_MAX_BLOCKS)))
        with mic.lock:
            mic.queue = audio_queue
            mic.queue_max_blocks = audio_queue.maxsize

        def _on_audio(indata, frames, time_info, status_flags) -> None:
            """音频回调：只做拷贝入队 + 电平统计。推理在 worker 线程，绝不在回调里跑。"""
            try:
                if status_flags:
                    with mic.lock:
                        mic.input_flags += 1
                block = indata.reshape(-1).copy()
                # 诊断：只算电平，放在入队之前但绝不影响入队（异常内部吞掉）
                _update_level(mic, block)
                audio_queue.put_nowait(block)
            except queue.Full:
                # 有界队列满：丢弃本块，并标记需要重同步（绝不把缺口拼进同一窗口）
                with mic.lock:
                    mic.dropped_blocks += 1
                    mic.dropped_samples += MIC_BLOCK_SAMPLES
                    mic.pending_resync = True
            except Exception as exc:  # 回调异常绝不能让 PortAudio 崩掉进程
                with mic.lock:
                    mic.callback_errors += 1
                    if mic.fatal is None:
                        mic.fatal = f"音频回调异常：{type(exc).__name__}: {exc}"

        def _on_finished() -> None:
            with mic.lock:
                mic.finished = True

        stream = _open_microphone(
            device_index=int(session.device_index),
            sample_rate=MIC_SAMPLE_RATE,
            blocksize=MIC_BLOCK_SAMPLES,
            callback=_on_audio,
            finished_callback=_on_finished,
        )
        session.stream = stream

        with self._lock:
            if stop_event.is_set():
                session.stop_reason = "stopped_during_loading"
                return
            session.recognizer = recognizer
            session.listening_started_mono = time.monotonic()
            session.listening_started_at = datetime.now(timezone.utc)
            session.opened = True
            self._config = stack.config() | {
                "input_sample_rate": MIC_SAMPLE_RATE,
                "channels": MIC_CHANNELS,
                "dtype": MIC_DTYPE,
                "block_samples": MIC_BLOCK_SAMPLES,
                "queue_max_blocks": audio_queue.maxsize,
            }
            self._state = "listening"

        logger.info(
            "麦克风已打开：device=%s idx=%s hostapi=%s sr=%s channels=%s dtype=%s block=%s",
            session.device_name, session.device_index, session.device_hostapi,
            MIC_SAMPLE_RATE, MIC_CHANNELS, MIC_DTYPE, MIC_BLOCK_SAMPLES,
        )

        while not stop_event.is_set():
            with mic.lock:
                fatal = mic.fatal
            if fatal:
                raise MicrophoneUnavailable(fatal)

            try:
                block = audio_queue.get(timeout=QUEUE_GET_TIMEOUT)
            except queue.Empty:
                if stop_event.is_set():
                    break
                if not self._stream_alive(session):
                    raise AudioDeviceLost(
                        "音频输入流已停止：设备可能被拔出、被其它程序独占或驱动异常。"
                    )
                continue

            self._resync(session, recognizer)
            self._shadow_push(acoustic, block, session)
            records = recognizer.push(block)
            self._sync(recognizer, records)

        session.stop_reason = session.stop_reason or "stopped"

    def _resync(self, session: _Session, recognizer) -> bool:
        """丢掉缺口两侧的音频，避免把不连续的数据拼进同一个窗口。

        * 清空窗口缓冲与 3 帧判定缓存（`WarmupGate` 不动：预热一个会话只做一次）。
        * 把丢弃的样本数累加进 `offset_s`，事件 time_s 加偏移后仍然单调。
        """
        mic = session.mic
        with mic.lock:
            if not mic.pending_resync:
                return False
            gap_samples = mic.dropped_samples
            mic.dropped_samples = 0
            mic.pending_resync = False
            mic.overflow_events += 1
            mic.resync_count += 1
            mic.offset_s += gap_samples / float(MIC_SAMPLE_RATE)

        try:
            recognizer.buffer.reset()
            recognizer.judge.reset()
            if getattr(recognizer, "resampler", None) is not None:
                recognizer.resampler.reset()
        except Exception as exc:  # 理论上不会发生；发生了也不能让采集线程崩
            logger.error("重同步清理识别缓存失败：%s", exc)
        self._reset_acoustic(session, "resync")
        logger.warning(
            "队列溢出：丢弃 %d 个块（%.3f s），已清空窗口缓冲与判定缓存",
            mic.dropped_blocks, gap_samples / float(MIC_SAMPLE_RATE),
        )
        return True

    def _make_acoustic(self, stack, session: _Session):
        """创建影子检测器；失败只记录，不影响原识别通路。"""
        factory = getattr(stack, "new_acoustic", None)
        if not callable(factory):
            return None
        try:
            detector = factory(input_rate=MIC_SAMPLE_RATE)
            session.acoustic = detector
            with self._lock:
                self._acoustic_session_resets += 1
                self._acoustic_state = detector.snapshot()
            return detector
        except Exception as exc:
            with self._lock:
                self._acoustic_errors += 1
                self._acoustic_last_error = (
                    f"{type(exc).__name__}: {exc}"
                )
            logger.error("声学影子检测器初始化失败，原识别继续：%s", exc)
            return None

    def _shadow_push(self, detector, block, session: _Session) -> None:
        """旁路处理当前音频块；任何影子异常都不冒泡到识别 worker。"""
        if detector is None:
            return
        try:
            candidates = detector.push(block)
            snapshot = detector.snapshot()
            offset_s = 0.0
            if session.source == "microphone":
                with session.mic.lock:
                    offset_s = session.mic.offset_s
            views = [
                AcousticCandidateView(
                    candidate_type=candidate.candidate_type,
                    start_time_s=round(candidate.start_time_s + offset_s, 3),
                    time_s=round(candidate.time_s + offset_s, 3),
                    duration_s=candidate.duration_s,
                    relative_db=candidate.relative_db,
                    reason=candidate.reason,
                    features=dict(candidate.features),
                )
                for candidate in candidates
            ]
            with self._lock:
                for view in views:
                    cluster_id = (
                        str(view.features.get("cluster_id", ""))
                        if view.candidate_type == "knock" else ""
                    )
                    replaced = False
                    if cluster_id:
                        for index, existing in enumerate(
                            self._acoustic_candidates
                        ):
                            if (
                                existing.candidate_type == "knock"
                                and existing.features.get("cluster_id")
                                == cluster_id
                            ):
                                self._acoustic_candidates[index] = view
                                replaced = True
                                break
                    if not replaced:
                        self._acoustic_candidates.append(view)
                self._acoustic_state = snapshot
            # 影子视图更新完之后再桥接落库：候选展示不依赖数据库成功与否。
            self._bridge_knock(session, views, snapshot)
        except Exception as exc:
            with self._lock:
                first_error = self._acoustic_errors == 0
                self._acoustic_errors += 1
                self._acoustic_last_error = (
                    f"{type(exc).__name__}: {exc}"
                )
            if first_error:
                logger.error("声学影子检测失败，原识别继续：%s", exc)

    def _bridge_knock(self, session: _Session, views, snapshot) -> None:
        """把本次新确认的敲门簇写成正式事件（每个簇只写一次）。

        去重键 = ``(session_id, 检测器重置代数, cluster_id)``：

        * `session_id`：监听 stop→start 会换新 id，同一 `knock-1` 是新会话的新簇；
        * `reset_count`：检测器 `reset()`（重同步/重建）后 `cluster_id` 从 1 重新
          计数，代数一变，旧簇不会"吃掉"新簇；
        * `cluster_id`：区分同一代内的不同簇。

        本方法**绝不抛出**：任何异常都只累计到 `_knock_persist_errors`，
        worker 线程与监听状态机不受影响。
        """
        if not self._knock_persist:
            return
        if session.source != "microphone":
            return
        if not views:
            return

        generation = int(snapshot.get("reset_count", 0) or 0)
        pending: List[tuple[tuple[str, int, str], datetime]] = []
        with self._lock:
            for view in views:
                if view.candidate_type != "knock":
                    continue
                cluster_id = str(view.features.get("cluster_id", "") or "")
                if not cluster_id:
                    # 没有簇标识就无法去重，宁可不写，也不冒重复的风险。
                    continue
                key = (session.session_id, generation, cluster_id)
                if key in self._knock_persisted:
                    # 同一簇已写成功：后续更新只刷新内存视图，不重复写库。
                    continue
                if key in self._knock_inflight:
                    # 同一次调用里重复出现（或极端并发）：已有一次在写，跳过。
                    continue
                # 只标记"进行中"；写成功后才转入"已完成"，失败即撤销，
                # 下一个候选更新会重新尝试（见下）。
                self._knock_inflight.add(key)
                pending.append((key, self._event_time_for(session, view)))

        for key, occurred_at in pending:
            try:
                event = _persist_knock_event(occurred_at)
            except Exception as exc:
                with self._lock:
                    self._knock_inflight.discard(key)
                    self._knock_persist_errors += 1
                    self._knock_persist_last_error = f"{type(exc).__name__}: {exc}"
                logger.error("敲门事件写入数据库失败（监听继续，允许重试）：%s", exc)
                continue
            with self._lock:
                self._knock_inflight.discard(key)
                self._knock_persisted.add(key)
                self._knock_events_persisted += 1
                self._knock_last_event_id = getattr(event, "id", None)

    @staticmethod
    def _event_time_for(session: _Session, view) -> datetime:
        """把候选的 ``start_time_s`` 还原成带 UTC 时区的事件时间。

        `view.start_time_s` 在 `_shadow_push` 里**已经加过** `offset_s`（丢块
        补偿），所以这里只做一次锚点平移，绝不再加偏移，否则丢过块的会话会
        被重复累加。锚点取 `listening_started_at`（真正开始采集的 UTC 时刻），
        不是 `started_at`（会话创建，早一个装栈周期）。
        """
        anchor = session.listening_started_at
        if anchor is None:  # 理论上不会发生；退回当前时间，至少保证带时区
            return datetime.now(timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        return anchor + timedelta(seconds=float(view.start_time_s))

    def _reset_acoustic(self, session: _Session, reason: str) -> None:
        detector = session.acoustic
        if detector is None:
            return
        try:
            detector.reset(reason=reason)
            snapshot = detector.snapshot()
        except Exception as exc:
            with self._lock:
                self._acoustic_errors += 1
                self._acoustic_last_error = (
                    f"{type(exc).__name__}: {exc}"
                )
            logger.error("声学影子检测器重置失败：%s", exc)
            return
        with self._lock:
            if reason == "resync":
                self._acoustic_resync_resets += 1
            self._acoustic_state = snapshot

    # -- 结果合并 -----------------------------------------------------------
    def _sync(self, recognizer, records) -> None:
        """把本块的判定结果与计数并入服务状态（锁内）。"""
        now = datetime.now(timezone.utc)
        stats = getattr(recognizer, "stats", None)
        with self._lock:
            session = self._session
            offset_s = 0.0
            offset_samples = 0
            if session is not None and session.source == "microphone":
                with session.mic.lock:
                    offset_s = session.mic.offset_s
                offset_samples = int(round(offset_s * MIC_TARGET_RATE))

            for record in records or ():
                smoothed = getattr(record, "smoothed", None) or {}
                raw = getattr(record, "per_group", None) or {}
                triggers = getattr(record, "triggers", ()) or ()
                for trigger in triggers:
                    self._seq += 1
                    self._triggers += 1
                    self._events.append(
                        ListenEvent(
                            seq=self._seq,
                            event_type=trigger.group,
                            score=float(trigger.score),
                            occurred_at=now,
                            time_s=float(trigger.time_s) + offset_s,
                            frame_index=int(trigger.frame_index),
                        )
                    )
                # 诊断：无论是否触发都记录该帧的分数（只读，不影响上面的判定）
                self._frame_scores.append(
                    FrameScore(
                        frame_index=int(getattr(record, "frame_index", -1)),
                        window_index=int(getattr(record, "window_index", -1)),
                        time_s=round(float(getattr(record, "time_s", 0.0)) + offset_s, 3),
                        doorbell=_opt_float(smoothed.get("doorbell")),
                        knock=_opt_float(smoothed.get("knock")),
                        raw_doorbell=_opt_float(raw.get("doorbell")),
                        raw_knock=_opt_float(raw.get("knock")),
                        triggered=bool(triggers),
                    )
                )
                for group, value in smoothed.items():
                    peak = _opt_float(value)
                    if peak is not None and peak > self._peak_smoothed.get(str(group), 0.0):
                        self._peak_smoothed[str(group)] = peak
            if stats is not None:
                self._frames = FrameStats(
                    produced=int(getattr(stats, "frames_produced", 0)),
                    scored=int(getattr(stats, "frames_scored", 0)),
                    warmup_dropped=int(getattr(stats, "warmup_frames_dropped", 0)),
                )
                self._stream_pos = int(getattr(stats, "stream_pos", 0)) + offset_samples
                self._buffered = int(getattr(stats, "buffered", 0))


#: 进程内单例。构造过程不加载任何识别依赖、不打开任何设备。
listen_service = ListenService(
    persist=PERSIST_ENABLED, knock_persist=KNOCK_PERSIST_ENABLED
)
