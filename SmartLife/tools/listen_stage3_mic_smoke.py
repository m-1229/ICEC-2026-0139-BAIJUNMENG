#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 3：真实麦克风采集通路的**无硬件**自动化测试。

运行环境（必须是原型 venv，因为要加载 numpy/TF/sounddevice）
-------------------------------------------------------------
    yamnet_prototype\\.venv\\Scripts\\python.exe -B tools\\listen_stage3_mic_smoke.py

设计原则
--------
1. **不打开任何真实音频端点**：`app.recognition` 里留了两个接入点
   `_check_input_settings` / `_open_microphone`，本脚本把它们替换成假实现，
   于是整条采集链路（回调 -> 有界队列 -> worker -> 识别 -> 判定）都能在
   没有麦克风的机器上跑通，也不会录音。
2. 真实麦克风只在最后一步由人工/单独脚本验证，本脚本不碰硬件。
3. 只读数据库：前后比对 `sound_events` / `alert_settings` 行数。
4. 只判断行为与计数，不调整阈值、窗口、预热与类别组。

覆盖项
------
G  设备解析与参数校验（未指定/指定/越界/被排除/做不到 48k/开关关闭）
M  真实开流语义（假流）：listening、device.opened、真实设备信息、预热 2 帧、触发
O  队列溢出（有界队列）与时间轴重同步
D  故障：回调异常、设备断开、开流失败、装栈期 stop、stop 超时
C  约束：persist=False、无数据库引用、无音频文件、线程清理
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from listen_api_smoke import (  # noqa: E402  复用阶段 2 的 HTTP/服务/断言工具
    DB_PATH,
    PROTOTYPE_DIR,
    R,
    db_counts,
    free_port,
    get_status,
    http,
    start_server,
    stop_server,
    wait_state,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
SCAN_DIRS = (
    PROJECT_ROOT / "yamnet_prototype" / "recordings",
    PROJECT_ROOT / "yamnet_prototype" / "recordings_negative",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "static",
)


# --------------------------------------------------------------------------- #
# 假输入流（替代 sounddevice.InputStream）
# --------------------------------------------------------------------------- #
class _Sink:
    """feeder 用它往外送块：`block()` 正常块，`raw()` 故意送坏对象。"""

    def __init__(self, stream: "FakeStream") -> None:
        self.stream = stream

    def block(self, samples) -> None:
        stream = self.stream
        if not stream.active:
            return
        stream.calls += 1
        stream.callback(samples.reshape(-1, 1), int(samples.size), None, None)

    def raw(self, obj) -> None:
        stream = self.stream
        if not stream.active:
            return
        stream.calls += 1
        stream.callback(obj, 0, None, None)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeStream:
    """最小 InputStream 替身：属性名与 sounddevice 保持一致。"""

    def __init__(self, feeder, *, realtime: float = 0.02, block: int = 960):
        self.feeder = feeder
        self.realtime = realtime
        self.block = block
        self.active = False
        self.stopped = False
        self.closed = False
        self.calls = 0
        self.callback = None
        self.finished_callback = None
        self.device_index = None
        self.samplerate = 48000
        self.channels = 1
        self.dtype = "float32"
        self._thread: threading.Thread | None = None

    # -- 生命周期 -----------------------------------------------------------
    def start(self) -> None:
        self.active = True
        self._thread = threading.Thread(
            target=self._run, name="fake-mic-feeder", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        sink = _Sink(self)
        try:
            if self.realtime > 0:
                def paced_sleep(seconds: float) -> None:
                    time.sleep(min(seconds, self.realtime))
                sink.sleep = paced_sleep  # type: ignore[method-assign]
            self.feeder(sink)
        except Exception:
            pass

    def stop(self) -> None:
        self.active = False
        self.stopped = True

    def close(self) -> None:
        self.active = False
        self.closed = True


# --------------------------------------------------------------------------- #
# 假识别栈（只在需要制造异常/卡死时用，避免反复加载模型）
# --------------------------------------------------------------------------- #
class _FakeStats:
    def __init__(self) -> None:
        self.frames_produced = 0
        self.frames_scored = 0
        self.warmup_frames_dropped = 0
        self.stream_pos = 0
        self.buffered = 0


class FakeRecognizer:
    def __init__(self, *, sleep: float = 0.0, raises: bool = False):
        self.stats = _FakeStats()
        self.sleep = sleep
        self.raises = raises
        self.buffer = types.SimpleNamespace(reset=lambda: None)
        self.judge = types.SimpleNamespace(reset=lambda: None)
        self.resampler = None

    def reset_session(self) -> None:
        return None

    def push(self, block):
        if self.raises:
            raise RuntimeError("测试注入：推理崩溃")
        if self.sleep:
            time.sleep(self.sleep)
        self.stats.stream_pos += int(block.size)
        return []

    def flush(self):
        return []


class FakeStack:
    def __init__(self, recognizer_factory):
        self._factory = recognizer_factory
        self.created = []

    def config(self):
        return {"window_samples": 15600, "hop_samples": 7680}

    def new_recognizer(self, input_rate: int = 48000):
        rec = self._factory()
        self.created.append(rec)
        return rec

    def load_clip(self, clip):
        import numpy as np

        return np.zeros(48000, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def recognition_module():
    import app.recognition as module

    return module


class Patch:
    """上下文管理器：替换 app.recognition 里的接入点，退出时还原。"""

    def __init__(self, **replacement):
        self.replacement = replacement
        self.saved = {}

    def __enter__(self):
        module = recognition_module()
        for name, value in self.replacement.items():
            self.saved[name] = getattr(module, name)
            setattr(module, name, value)
        return module

    def __exit__(self, *_exc):
        module = recognition_module()
        for name, value in self.saved.items():
            setattr(module, name, value)
        return False


def fake_microphone(feeder_factory, *, realtime: float = 0.02, check=None,
                    open_error: Exception | None = None, created=None):
    """构造 _open_microphone 的假实现；created 收集打开的流对象。"""
    streams = created if created is not None else []

    def _open(device_index, sample_rate, blocksize, callback, finished_callback):
        if open_error is not None:
            raise open_error
        stream = FakeStream(feeder_factory(), realtime=realtime, block=blocksize)
        stream.callback = callback
        stream.finished_callback = finished_callback
        stream.device_index = device_index
        stream.samplerate = sample_rate
        streams.append(stream)
        stream.start()
        return stream

    checker = check if check is not None else (lambda *a, **k: None)
    return _open, checker, streams


def audio_snapshot() -> dict[str, tuple[int, float]]:
    """记录项目内所有音频文件的 (大小, mtime)，用于证明"没有产生录音"。"""
    snapshot: dict[str, tuple[int, float]] = {}
    for base in SCAN_DIRS:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                stat = path.stat()
                snapshot[str(path)] = (stat.st_size, round(stat.st_mtime, 3))
    return snapshot


def start_mic(base: str, **body):
    payload = {"source": "microphone"}
    payload.update(body)
    return http("POST", f"{base}/api/listen/start", payload)


def wait_listening(base: str, timeout: float = 90.0):
    return wait_state(base, ("listening", "error", "idle"), timeout=timeout)


def listening_workers() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("listen-worker")]


def load_48k_wav(relative: str):
    """读一段 48 kHz 素材，返回 (float32 单声道, 时长)。"""
    import numpy as np

    if str(PROTOTYPE_DIR) not in sys.path:
        sys.path.insert(0, str(PROTOTYPE_DIR))
    import yamnet_batch_eval as be

    data, info = be.read_wav(str(PROTOTYPE_DIR / relative))
    assert int(info["sample_rate"]) == 48000, (relative, info["sample_rate"])
    mono = data.mean(axis=1) if info["channels"] > 1 else data[:, 0]
    mono = np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)
    return mono, mono.size / 48000.0


def silent_then(mono, lead_seconds: float, block: int = 960):
    """先送 lead_seconds 秒静音，再送素材（模拟"流已在跑，事件稍后发生"）。"""

    def feeder(sink):
        import numpy as np

        zeros = np.zeros(int(lead_seconds * 48000), dtype=np.float32)
        for i in range(0, zeros.size, block):
            sink.block(zeros[i:i + block])
            sink.sleep(0.02)
        for i in range(0, mono.size, block):
            sink.block(mono[i:i + block])
            sink.sleep(0.02)

    return feeder


# --------------------------------------------------------------------------- #
# G：设备解析与参数校验（不加载模型、不打开流）
# --------------------------------------------------------------------------- #
def group_devices(base: str, real_devices: list[dict]) -> None:
    print("\n[G] 设备解析与参数校验")

    code, listing = http("GET", f"{base}/api/listen/devices")
    target = listing.get("target") or {}
    R.add("G1", "devices 仍只枚举", "200 + available=true + 有 devices",
          f"{code} available={listing.get('available')} n={len(listing.get('devices') or [])}",
          code == 200 and listing.get("available") is True and len(listing.get("devices") or []) > 0)
    R.add("G2", "推荐设备为 48k / 未被排除", "sample_rate=48000 excluded=false",
          f"{target.get('name')} / {target.get('default_samplerate')} / "
          f"idx={target.get('index')} excluded={target.get('excluded')}",
          bool(target) and float(target.get("default_samplerate") or 0) == 48000.0
          and target.get("excluded") is False)

    code, payload = start_mic(base, device=99999)
    R.add("G3", "越界设备索引被拒", "400",
          f"{code} {str(payload)[:70]}", code == 400)

    excluded = next((d for d in real_devices if d.get("excluded")), None)
    if excluded is None:
        R.info("G4", "被排除设备拒绝", "本机没有可测试的被排除设备")
    else:
        code, payload = start_mic(base, device=excluded["index"])
        R.add("G4", "被排除设备（虚拟/回环）被拒", "400",
              f"{code} idx={excluded['index']} {str(payload)[:50]}", code == 400)

    def _reject_48k(*_a, **_k):
        from app.recognition import DeviceNotAllowed

        raise DeviceNotAllowed(
            "设备 1 不支持 48000 Hz / 1 声道 / float32：PortAudioError: Invalid sample rate"
            "（不做重采样降级）"
        )

    opened: list = []
    open_fn, _, _ = fake_microphone(lambda: None, created=opened)
    with Patch(_check_input_settings=_reject_48k, _open_microphone=open_fn):
        code, payload = start_mic(base)
        R.add("G5", "不支持 48kHz 的设备被拒", "400 且提示不支持 48000",
              f"{code} {str(payload)[:80]}",
              code == 400 and "48000" in str(payload))
        R.add("G6", "被拒时不打开输入流", "open 调用 0 次", f"{len(opened)} 次",
              len(opened) == 0)

    with Patch(ALLOW_MICROPHONE=False):
        code, payload = start_mic(base)
        R.add("G7", "麦克风开关关闭时拒绝", "400",
              f"{code} {str(payload)[:60]}", code == 400)

    # 真实 check_input_settings（只查询驱动能力，不打开流）——记录即可
    module = recognition_module()
    if module.sounddevice_available() and target:
        try:
            module._check_input_settings(int(target["index"]))
            R.info("G8", "真实 check_input_settings(48000/1/float32)",
                   f"通过（idx={target['index']}）")
        except Exception as exc:
            R.info("G8", "真实 check_input_settings(48000/1/float32)",
                   f"未通过：{type(exc).__name__}: {str(exc)[:60]}")
    else:
        R.info("G8", "真实 check_input_settings", "sounddevice 不可用或无推荐设备")

    code, status = http("GET", f"{base}/api/listen/status")
    R.add("G9", "参数校验失败后仍为 idle", "idle",
          f"{status['state']} opened={status['device']['opened']}",
          status["state"] == "idle")


# --------------------------------------------------------------------------- #
# M：真实开流语义（假流，真模型）
# --------------------------------------------------------------------------- #
def group_pipeline(base: str, target: dict) -> None:
    print("\n[M] 麦克风通路（假流 + 真模型）：listening / device.opened / 预热 / 触发")

    mono, duration = load_48k_wav("recordings/门铃/doorbell_large.wav")
    created: list[FakeStream] = []
    open_fn, checker, _ = fake_microphone(
        lambda: silent_then(mono, 2.0), realtime=0.02, created=created
    )

    with Patch(_open_microphone=open_fn, _check_input_settings=checker):
        code, started = start_mic(base)
        R.add("M1", "microphone start 返回 202", "202 + loading/listening",
              f"{code} {started.get('state')}", code == 202)
        R.add("M2", "start 响应 source=microphone", "microphone",
              str(started.get("source")), started.get("source") == "microphone")
        R.add("M3", "start 时尚未标记 opened", "opened=false",
              f"{started.get('device', {}).get('opened')}",
              started.get("device", {}).get("opened") is False)

        status, seen = wait_state(base, ("listening", "error", "idle"), timeout=120)
        R.add("M4", "状态进入 listening", "listening",
              f"{status['state']} 序列={seen}", status["state"] == "listening")

        device = status["device"]
        R.add("M5", "device.opened=true", "True", str(device["opened"]), device["opened"] is True)
        R.add("M6", "设备信息为真实值", "index/name/hostapi 非空且与推荐项一致",
              f"idx={device['index']} {device['name'][:18]} {device['hostapi']}",
              device["index"] == target["index"] and device["name"] == target["name"]
              and device["hostapi"] == target["hostapi"])
        R.add("M7", "采样率/声道/dtype", "48000 / 1 / float32",
              f"{device['sample_rate']} / {device['channels']} / {device['dtype']}",
              device["sample_rate"] == 48000 and device["channels"] == 1
              and device["dtype"] == "float32")
        R.add("M8", "config 记录采集参数", "input_sample_rate=48000",
              f"{status['config'].get('input_sample_rate')} "
              f"block={status['config'].get('block_samples')}",
              status["config"].get("input_sample_rate") == 48000)
        R.add("M9", "队列有界", "queue_max_blocks>0",
              f"{status['mic']['queue_max_blocks']}", status["mic"]["queue_max_blocks"] > 0)

        # 让素材跑完（2 s 静音 + 素材时长）
        deadline = time.time() + duration + 15
        best = None
        while time.time() < deadline:
            status = get_status(base)
            if status["events"]:
                best = max(status["events"], key=lambda e: e["score"])
                break
            if status["state"] != "listening":
                break
            time.sleep(0.1)
        status = get_status(base)
        events = status["events"]
        R.add("M10", "真实素材经麦克风通路触发", ">=1 条 doorbell",
              f"{len(events)} 条 {[e['event_type'] for e in events][:2]}",
              any(e["event_type"] == "doorbell" for e in events))
        if events:
            R.info("M11", "最高分与首次触发",
                   f"{best['event_type']} {best['score']:.4f} @ {best['time_s']:.2f}s")
        else:
            R.info("M11", "未触发", f"帧数={status['frames']}")

        frames = status["frames"]
        acoustic = status["acoustic"]
        R.add("M10a", "声学影子状态已上报且仅作影子",
              "shadow_only=true",
              f"shadow_only={acoustic['shadow_only']} "
              f"frames={acoustic['frames_processed']}",
              acoustic["shadow_only"] is True
              and acoustic["frames_processed"] > 0)
        R.add("M10b", "门铃素材产生门铃影子候选",
              ">=1 条 doorbell 候选",
              f"{len(acoustic['candidates'])} 条 "
              f"{[c['candidate_type'] for c in acoustic['candidates'][:3]]}",
              any(c["candidate_type"] == "doorbell"
                  for c in acoustic["candidates"]))
        R.add("M12", "预热丢弃 2 帧", "warmup_dropped=2",
              f"{frames['warmup_dropped']}", frames["warmup_dropped"] == 2)
        R.add("M13", "帧计数自洽（produced - 预热 == scored）",
              "produced>=3 且 scored = produced-2",
              f"produced={frames['produced']} scored={frames['scored']}",
              frames["produced"] >= 3
              and frames["scored"] == frames["produced"] - frames["warmup_dropped"])
        R.add("M14", "stream_pos 与时间轴一致", "stream_pos ≈ 已采集样本",
              f"stream_pos={status['stream_pos']} buffered={status['buffered']}",
              status["stream_pos"] > 0 and status["buffered"] >= 0)

        # 重复 start
        code, conflict = start_mic(base)
        R.add("M15", "运行中重复 start 被拒", "409",
              f"{code} {str(conflict)[:40]}", code == 409)

        # stop：必须真的关闭流并让线程退出
        code, stopped = http("POST", f"{base}/api/listen/stop")
        R.add("M16", "stop 回到 idle", "200 idle",
              f"{code} {stopped.get('state')}", code == 200 and stopped.get("state") == "idle")
        R.add("M17", "stop 后 device.opened=false", "False",
              str(stopped["device"]["opened"]), stopped["device"]["opened"] is False)
        time.sleep(0.3)
        stream = created[0] if created else None
        R.add("M18", "InputStream 被 stop+close", "stopped=True closed=True",
              f"stopped={getattr(stream, 'stopped', None)} closed={getattr(stream, 'closed', None)}",
              bool(stream and stream.stopped and stream.closed))
        R.add("M19", "worker 线程已退出", "0 个 listen-worker",
              f"{listening_workers()}", not listening_workers())

        code, again = http("POST", f"{base}/api/listen/stop")
        R.add("M20", "重复 stop 幂等", "200 idle",
              f"{code} {again.get('state')}", code == 200 and again.get("state") == "idle")

        # 新会话重置
        seq_before = [e["seq"] for e in stopped["events"]]
        session_before = stopped.get("session_id")
        created.clear()
        stalled: list[FakeStream] = []
        open_fn2, checker2, _ = fake_microphone(
            lambda: silent_then(mono[: 48000 * 3], 1.0), realtime=0.02, created=stalled
        )
        with Patch(_open_microphone=open_fn2, _check_input_settings=checker2):
            code, second = start_mic(base)
            status2, _ = wait_state(base, ("listening", "error"), timeout=120)
            time.sleep(2.6)          # 攒够 4 帧：预热 2 帧 + 至少 1 帧送判定
            status2 = get_status(base)
            R.add("M21", "重新 start 生成新会话", "session_id 变化",
                  f"{session_before} -> {status2['session_id']}",
                  status2["session_id"] not in (None, session_before))
            R.add("M21a", "新会话重置声学影子状态",
                  "session_resets=1 且 reset_count=0",
                  f"session_resets={status2['acoustic']['session_resets']} "
                  f"reset_count={status2['acoustic']['reset_count']}",
                  status2["acoustic"]["session_resets"] == 1
                  and status2["acoustic"]["reset_count"] == 0)
            R.add("M22", "新会话预热重新计 2", "warmup_dropped=2",
                  f"{status2['frames']['warmup_dropped']}",
                  status2["frames"]["warmup_dropped"] == 2)
            seqs = [e["seq"] for e in status2["events"]]
            R.add("M23", "事件序号按新会话重置", "seq 从 1 起连续",
                  f"{seqs[:4]} triggers={status2['triggers']}",
                  (seqs == list(range(1, len(seqs) + 1))) and status2["triggers"] == len(seqs))
            R.add("M24", "统计清零后重新累计", "produced >= 3",
                  f"produced={status2['frames']['produced']}",
                  status2["frames"]["produced"] >= 3)
            http("POST", f"{base}/api/listen/stop")
            wait_state(base, ("idle", "error"), timeout=60)


# --------------------------------------------------------------------------- #
# S：声学特征影子（假流 + 真模型，不碰硬件）
# --------------------------------------------------------------------------- #
def group_acoustic_shadow(base: str) -> None:
    print("\n[S] 声学特征影子：合成敲门候选 / 状态接口 / 正式事件隔离")

    import numpy as np

    def synthetic_knocks(sink):
        n = 960
        t = np.arange(n, dtype=np.float32) / 48000.0
        env = np.exp(-t / 0.045).astype(np.float32)
        one = (
            (
                np.sin(2 * np.pi * 180 * t)
                + 0.45 * np.sin(2 * np.pi * 420 * t)
            ) * env * 0.20
        ).astype(np.float32)
        for _ in range(100):
            sink.block(np.zeros(960, dtype=np.float32))
            sink.sleep(0.02)
        for _ in range(3):
            for i in range(0, one.size, 960):
                sink.block(one[i:i + 960])
                sink.sleep(0.02)
            for _ in range(10):
                sink.block(np.zeros(960, dtype=np.float32))
                sink.sleep(0.02)
        for _ in range(50):
            sink.block(np.zeros(960, dtype=np.float32))
            sink.sleep(0.02)

    # 落库入口换成内存记录器：本脚本绝不能往正式数据库写测试事件。
    recorded: list = []

    def _record_knock(occurred_at):
        recorded.append(occurred_at)
        return types.SimpleNamespace(id=9000 + len(recorded))

    # 桥接开关现在是"默认关"的安全默认；本组显式开启专测桥接行为，
    # 结束后还原，避免影响后续用例（默认关的行为由 knock_bridge_smoke G 组覆盖）。
    _svc = recognition_module().listen_service
    saved_knock_flag = _svc._knock_persist
    _svc._knock_persist = True

    open_fn, checker, created = fake_microphone(
        lambda: synthetic_knocks, realtime=0.02
    )
    with Patch(_open_microphone=open_fn, _check_input_settings=checker,
               _persist_knock_event=_record_knock):
        code, _ = start_mic(base)
        status, _ = wait_state(base, ("listening", "error"), timeout=120)
        deadline = time.time() + 12
        while time.time() < deadline:
            status = get_status(base)
            acoustic = status["acoustic"]
            if any(
                c["candidate_type"] == "knock"
                and int(c["features"].get("impact_count", 0)) >= 3
                for c in acoustic["candidates"]
            ):
                break
            if status["state"] != "listening":
                break
            time.sleep(0.1)
        acoustic = status["acoustic"]
        knocks = [
            c for c in acoustic["candidates"]
            if c["candidate_type"] == "knock"
        ]
        R.add("S1", "合成连续冲击产生敲门影子候选",
              ">=1 条 knock 候选",
              f"{len(knocks)} 条 {knocks[:1]}", bool(knocks))
        R.add("S1a", "同一簇只显示一个并更新冲击数",
              "1 条 knock，impact_count=3",
              f"count={len(knocks)} "
              f"impacts={[c['features'].get('impact_count') for c in knocks]}",
              len(knocks) == 1
              and int(knocks[0]["features"].get("impact_count", 0)) == 3)
        R.add("S2", "候选包含时间、特征和原因",
              "time/features/reason 均存在",
              str(knocks[0] if knocks else None),
              bool(knocks) and knocks[0]["time_s"] > 0
              and bool(knocks[0]["features"]) and bool(knocks[0]["reason"]))
        R.add("S3", "影子候选不产生正式事件",
              "events=0",
              f"events={len(status['events'])}",
              len(status["events"]) == 0)
        status = get_status(base)
        persisted = status["acoustic"]
        R.add("S3b", "同一敲门簇只写一条正式事件（后续更新不重复）",
              "persisted=1 / errors=0",
              f"recorded={len(recorded)} "
              f"persisted_events={persisted['persisted_events']} "
              f"errors={persisted['persist_errors']}",
              len(recorded) == 1
              and persisted["persisted_events"] == 1
              and persisted["persist_errors"] == 0)
        R.add("S3c", "显式开启后桥接开关与 YAMNet 落库开关相互独立",
              "knock_persist_enabled=True, persist=False",
              f"{persisted['knock_persist_enabled']} / {status['persist']}",
              persisted["knock_persist_enabled"] is True
              and status["persist"] is False)
        R.add("S4", "原 YAMNet 帧计数继续推进",
              "produced>0",
              f"produced={status['frames']['produced']}",
              status["frames"]["produced"] > 0)
        code, stopped = http("POST", f"{base}/api/listen/stop")
        R.add("S5", "影子会话可正常停止", "200 idle",
              f"{code} {stopped.get('state')}",
              code == 200 and stopped.get("state") == "idle")
    _svc._knock_persist = saved_knock_flag

    R.add("S6", "影子测试没有真实开流",
          "使用 FakeStream",
          f"fake_streams={len(created)}",
          bool(created) and all(hasattr(x, "feeder") for x in created))


# --------------------------------------------------------------------------- #
# O：队列溢出与时间轴重同步
# --------------------------------------------------------------------------- #
def group_overflow(base: str) -> None:
    print("\n[O] 有界队列溢出 / 时间轴重同步")

    import numpy as np

    module = recognition_module()
    original_max = module.QUEUE_MAX_BLOCKS

    def flood(sink):
        block = np.zeros(960, dtype=np.float32)
        for _ in range(400):          # 一口气灌 400 块（8 s 音频），必然压满队列
            sink.block(block + 0.001)

    created: list[FakeStream] = []
    open_fn, checker, _ = fake_microphone(lambda: flood, realtime=0.0, created=created)

    module.QUEUE_MAX_BLOCKS = 4
    try:
        with Patch(_open_microphone=open_fn, _check_input_settings=checker):
            code, started = start_mic(base)
            status, _ = wait_state(base, ("listening", "error"), timeout=120)
            deadline = time.time() + 20
            while time.time() < deadline:
                status = get_status(base)
                if status["mic"]["overflow_events"] >= 1:
                    break
                if status["state"] != "listening":
                    break
                time.sleep(0.05)
            status = get_status(base)
            mic = status["mic"]
            R.add("O1", "队列上限生效", f"queue_max_blocks={4}",
                  f"{mic['queue_max_blocks']}", mic["queue_max_blocks"] == 4)
            R.add("O2", "发生溢出并记录丢块", "dropped_blocks>0",
                  f"{mic['dropped_blocks']}", mic["dropped_blocks"] > 0)
            R.add("O3", "触发重同步", "resync_count>=1 且 == overflow_events",
                  f"resync={mic['resync_count']} overflow={mic['overflow_events']}",
                  mic["resync_count"] >= 1 and mic["resync_count"] == mic["overflow_events"])
            R.add("O4", "时间轴偏移被累计", "timeline_offset_s>0",
                  f"{mic['timeline_offset_s']}", mic["timeline_offset_s"] > 0)
            R.add("O5", "溢出不影响会话存活", "listening",
                  f"{status['state']}", status["state"] == "listening")

            first = get_status(base)
            time.sleep(0.6)
            second = get_status(base)
            R.add("O6", "stream_pos 单调不倒退", "second >= first",
                  f"{first['stream_pos']} -> {second['stream_pos']}",
                  second["stream_pos"] >= first["stream_pos"])
            R.add("O7", "偏移量单调不倒退", "second >= first",
                  f"{first['mic']['timeline_offset_s']} -> "
                  f"{second['mic']['timeline_offset_s']}",
                  second["mic"]["timeline_offset_s"] >= first["mic"]["timeline_offset_s"])
            R.add("O9", "重同步重置声学影子状态",
                  "resync_resets>=1 且 last_reset_reason=resync",
                  f"resync_resets={second['acoustic']['resync_resets']} "
                  f"last_reset={second['acoustic']['last_reset_reason']}",
                  second["acoustic"]["resync_resets"] >= 1
                  and second["acoustic"]["last_reset_reason"] == "resync")

            code, stopped = http("POST", f"{base}/api/listen/stop")
            R.add("O8", "溢出后仍可正常停止", "200 idle",
                  f"{code} {stopped.get('state')}",
                  code == 200 and stopped.get("state") == "idle")
    finally:
        module.QUEUE_MAX_BLOCKS = original_max


# --------------------------------------------------------------------------- #
# D：故障注入
# --------------------------------------------------------------------------- #
def group_failures(base: str) -> None:
    print("\n[D] 故障：回调异常 / 设备断开 / 开流失败 / 装栈期 stop / stop 超时")

    import numpy as np

    module = recognition_module()

    # D1 回调异常
    def bad_feeder(sink):
        sink.raw(object())          # 没有 reshape，服务端回调必须自己兜住
        time.sleep(0.2)

    open_fn, checker, _ = fake_microphone(lambda: bad_feeder, realtime=0.0)
    with Patch(_open_microphone=open_fn, _check_input_settings=checker):
        start_mic(base)
        status, _ = wait_state(base, ("error", "listening"), timeout=90)
        deadline = time.time() + 10
        while time.time() < deadline and status["state"] == "listening":
            time.sleep(0.1)
            status = get_status(base)
        R.add("D1a", "回调异常被计数", "callback_errors>=1",
              f"{status['mic']['callback_errors']}", status["mic"]["callback_errors"] >= 1)
        R.add("D1b", "回调异常落 error", "error + 含回调异常",
              f"{status['state']} {str(status.get('last_error'))[:40]}",
              status["state"] == "error" and "回调异常" in str(status.get("last_error")))
        R.add("D1c", "异常后线程退出", "0 个 listen-worker",
              f"{listening_workers()}", not listening_workers())
        http("POST", f"{base}/api/listen/stop")

    # D2 设备断开（流 active=False 且不再有数据）
    def slow_feeder(sink):
        block = np.zeros(960, dtype=np.float32)
        for _ in range(5):
            sink.block(block)
            time.sleep(0.05)

    created: list[FakeStream] = []
    open_fn, checker, _ = fake_microphone(lambda: slow_feeder, realtime=0.0, created=created)
    with Patch(_open_microphone=open_fn, _check_input_settings=checker):
        start_mic(base)
        wait_state(base, ("listening",), timeout=90)
        time.sleep(0.4)
        if created:
            created[0].active = False        # 模拟拔出 / 被独占
        deadline = time.time() + 10
        status = get_status(base)
        while time.time() < deadline and status["state"] == "listening":
            time.sleep(0.1)
            status = get_status(base)
        R.add("D2a", "设备断开落 error", "error + stop_reason=device_lost",
              f"{status['state']} / {status.get('stop_reason')}",
              status["state"] == "error" and status.get("stop_reason") == "device_lost")
        R.add("D2b", "错误信息可读", "含'已停止'",
              str(status.get("last_error"))[:60],
              "已停止" in str(status.get("last_error")))
        http("POST", f"{base}/api/listen/stop")

    # D3 开流失败
    from app.recognition import MicrophoneUnavailable

    def boom(*_a, **_k):
        raise MicrophoneUnavailable("无法打开麦克风设备 14（48000 Hz）：PortAudioError: 设备被占用")

    with Patch(_open_microphone=boom, _check_input_settings=lambda *a, **k: None):
        start_mic(base)
        status, _ = wait_state(base, ("error", "listening"), timeout=90)
        R.add("D3", "开流失败落 error", "error + 含'无法打开'",
              f"{status['state']} {str(status.get('last_error'))[:45]}",
              status["state"] == "error" and "无法打开" in str(status.get("last_error")))
        http("POST", f"{base}/api/listen/stop")

    # D4 装栈期 stop（模型加载不可中断）
    original_loader = module._load_stack

    def slow_loader():
        time.sleep(1.5)
        return original_loader()

    module._load_stack = slow_loader
    try:
        with Patch(_open_microphone=lambda *a, **k: FakeStream(lambda s: None),
                   _check_input_settings=lambda *a, **k: None):
            code, started = start_mic(base)
            began = time.time()
            code, stopped = http("POST", f"{base}/api/listen/stop")
            R.add("D4a", "装栈期 stop 回到 idle", "200 idle",
                  f"{code} {stopped.get('state')}",
                  code == 200 and stopped.get("state") == "idle")
            R.add("D4b", "stop_reason=stopped_during_loading", "stopped_during_loading",
                  str(stopped.get("stop_reason")),
                  stopped.get("stop_reason") == "stopped_during_loading")
            R.info("D4c", "装栈期 stop 耗时", f"{time.time() - began:.2f}s（模型加载不可中断）")
    finally:
        module._load_stack = original_loader

    # D5 stop 超时：worker 卡在 push 里
    original_timeout = module.STOP_JOIN_TIMEOUT
    module.STOP_JOIN_TIMEOUT = 0.2

    def blocking_feeder(sink):
        block = np.zeros(960, dtype=np.float32)
        for _ in range(3):
            sink.block(block)
            time.sleep(0.05)
        time.sleep(3.0)

    slow_stack = FakeStack(lambda: FakeRecognizer(sleep=1.5))
    try:
        with Patch(_load_stack=lambda: slow_stack,
                   _open_microphone=fake_microphone(lambda: blocking_feeder, realtime=0.0)[0],
                   _check_input_settings=lambda *a, **k: None):
            start_mic(base)
            wait_state(base, ("listening", "error"), timeout=30)
            time.sleep(0.3)
            code, stopped = http("POST", f"{base}/api/listen/stop")
            R.add("D5a", "stop 超时不谎报 idle", "error",
                  f"{code} {stopped.get('state')}",
                  code == 200 and stopped.get("state") == "error")
            code_retry, _ = start_mic(base)
            R.add("D5b", "线程存活期间禁止再启动", "409",
                  f"{code_retry}", code_retry == 409)
            deadline = time.time() + 15
            while time.time() < deadline and listening_workers():
                time.sleep(0.1)
            status = get_status(base)
            R.add("D5c", "线程退出后仍保留 error", "error（不自动谎报 idle）",
                  f"{status['state']}", status["state"] == "error")
    finally:
        module.STOP_JOIN_TIMEOUT = original_timeout
        http("POST", f"{base}/api/listen/stop")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print(f"阶段 3 麦克风采集（无硬件）测试  解释器={sys.executable}")
    print(f"麦克风开关环境变量 SMARTLIFE_ENABLE_MICROPHONE="
          f"{os.environ.get('SMARTLIFE_ENABLE_MICROPHONE', '(未设置)')}")
    print("=" * 78)

    before_db = db_counts()
    before_files = audio_snapshot()

    port = free_port()
    server, thread, base = start_server(port)
    print(f"  服务已启动 {base}")

    try:
        code, health = http("GET", f"{base}/api/health")
        R.add("C0", "health 报告麦克风能力", "microphone_capture_available=true",
              f"{code} enabled={health.get('microphone_enabled')} "
              f"capture={health.get('microphone_capture_available')}",
              code == 200 and health.get("microphone_capture_available") is True)

        _, listing = http("GET", f"{base}/api/listen/devices")
        target = listing.get("target") or {}
        devices = listing.get("devices") or []

        if not target:
            R.info("G0", "无推荐设备", "本机没有可用输入设备，跳过设备相关用例")
        group_devices(base, devices)

        if target:
            group_pipeline(base, target)
        group_acoustic_shadow(base)
        group_overflow(base)
        group_failures(base)

        # ---- 约束 ----
        print("\n[C] 约束：persist / 数据库 / 音频文件 / 线程")
        status = get_status(base)
        R.add("C1", "persist 恒为 False", "False", str(status["persist"]),
              status["persist"] is False)

        src = (PROJECT_ROOT / "app" / "recognition.py").read_text(encoding="utf-8")
        src_api = (PROJECT_ROOT / "app" / "listen_api.py").read_text(encoding="utf-8")
        forbidden = [t for t in ("import database", "get_connection",
                                 "INSERT INTO", "create_simulated_event",
                                 "soundfile.write", "savetxt", "wave.open")
                     if t in src + src_api]
        # 敲门事件桥接：允许且**只允许**一个延迟落库入口。
        allowed = "from .database import create_acoustic_knock_event"
        other_db = [
            line.strip() for line in src.splitlines()
            if "from .database" in line and line.strip() != allowed
        ]
        lazy_ok = src.count(allowed) == 1
        R.add("C2", "识别层无落盘调用、数据库只有 1 个延迟敲门入口",
              "无落盘/无其它 DB 引用，且恰好 1 处延迟 import",
              "无" if not forbidden and lazy_ok and not other_db
              else f"命中 {forbidden} lazy={lazy_ok} other={other_db}",
              not forbidden and lazy_ok and not other_db)

        after_files = audio_snapshot()
        added = sorted(set(after_files) - set(before_files))
        changed = sorted(k for k in set(after_files) & set(before_files)
                         if after_files[k] != before_files[k])
        R.add("C3", "未产生/改动任何音频文件", "新增 0、改动 0",
              f"新增 {len(added)} 改动 {len(changed)}"
              + (f" {added[:2]}" if added else ""),
              not added and not changed)

        R.add("C4", "无残留 listen-worker", "0 个",
              f"{listening_workers()}", not listening_workers())

    finally:
        stop_server(server, thread)

    after_db = db_counts()
    print()
    R.add("DB1", "sound_events 行数不变", str(before_db[0]), str(after_db[0]),
          before_db[0] == after_db[0])
    R.add("DB2", "alert_settings 行数不变", str(before_db[1]), str(after_db[1]),
          before_db[1] == after_db[1])

    print("\n" + "=" * 78)
    n_pass = len(R.passed())
    n_fail = len(R.failures())
    print(f"PASS {n_pass}  FAIL {n_fail}  INFO {len(R.infos())}")
    if R.infos():
        print("\n仅记录项：")
        for tid, name, _e, actual, _ok, note in R.infos():
            print(f"  {tid:<6s} {name}: {actual} {note}")
    if n_fail:
        print("\n未通过项：")
        for tid, name, expected, actual, _ok, note in R.failures():
            print(f"  {tid:<6s} {name}: 期望 {expected} / 实测 {actual} {note}")
    else:
        print("\n未通过项：无")
    print("结论:", "PASS" if n_fail == 0 else "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
