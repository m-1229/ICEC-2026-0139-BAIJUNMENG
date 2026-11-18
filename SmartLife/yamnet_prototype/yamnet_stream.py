#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SmartLife 流式音频模块：采集块 -> 16 kHz 连续流 -> 定长窗口 -> 判定。

纯计算，不含麦克风、不含 HTTP、不写数据库、不读文件。只用 numpy。

四层职责
--------
StreamResampler48k  48000 Hz float32 连续流 -> 16000 Hz float32 连续流
                    （201 抽头 Kaiser-beta=8 sinc 低通 + 3:1 抽取，因果、带状态）
WindowBuffer        16000 Hz 连续流 -> [(k, window)]，窗口 k 的起点恒为 k*HOP
WarmupGate          会话开始时丢弃前 N 个窗口的识别结果（方案甲，N=2），只丢一次
StreamingRecognizer 以上三者 + FrameJudge，一行接一块采集数据

为什么必须自己做重采样
----------------------
逐块 / 逐窗调用 FFT 重采样会在块边界产生宽带咔嗒，实测门铃分数偏差 0.4333，
与阈值同量级，足以翻转判定。本模块的因果 FIR 带内部状态，分块大小对输出
逐样本零影响（实测 max diff 1.7e-18，纯浮点舍入）。

为什么抽取相位必须跟全局计数
----------------------------
抽取条件是"全局输入样本序号 % 3 == 0"，而不是"本次块的局部序号"。否则块与块
之间的相位会漂移，同一段音频换个 chunk 大小就得到不同的波形。

常量的唯一来源（本模块不复制任何定义）
--------------------------------------
WINDOW_SAMPLES / HOP_SAMPLES / TARGET_SR / HOP_SECONDS / DECISION_LATENCY_S 来自
yamnet_recognizer；THETA_DEFAULT / SMOOTH_N / GROUPS 来自 yamnet_doorbell_eval。
改窗口、改阈值、改类别组，只改这两个文件，本模块自动跟随。

已知延迟（不补偿）
------------------
因果 FIR 有 (taps-1)/2 个输入样本的群延迟，48 kHz 下 = 100 样本 = 2.083 ms，
折合 16 kHz 输出 33.3 样本（0.033 个窗口）。它只会让所有判定整体后移 2 ms，
不改变任何分数；补偿需要未来样本，实时链路做不到，故保留并在此声明。

首帧时序（方案甲，warmup=2）
----------------------------
    第 1 个窗口产生     t = WINDOW/16000
    预热丢弃结束        t = (HOP + WINDOW)/16000
    首次完成 3 帧判定   t = (4*HOP + WINDOW)/16000
具体秒数见 yamnet_stream_selftest.py 的 A4，那里按公式打印并断言，不在这里硬编码。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 规则与窗口：全部 import，本文件不出现任何阈值/窗口/类别字面量。
from yamnet_recognizer import (  # noqa: E402
    DECISION_LATENCY_S,
    HOP_SAMPLES,
    HOP_SECONDS,
    TARGET_SR,
    WINDOW_SAMPLES,
    FrameJudge,
    FrameRecord,
    Trigger,
    _ModelBundle as ModelBundle,
    group_scores,
    score_window,
)
from yamnet_doorbell_eval import GROUPS, SMOOTH_N, THETA_DEFAULT  # noqa: E402

# --------------------------------------------------------------------------- #
# 别名（只做转发，不新增数值定义）
# --------------------------------------------------------------------------- #
WINDOW = WINDOW_SAMPLES          # 一个窗口的 16 kHz 样本数
HOP = HOP_SAMPLES                # 相邻窗口起点的间隔（16 kHz 样本）
WINDOW_SECONDS = WINDOW / TARGET_SR

# --------------------------------------------------------------------------- #
# 重采样参数（本模块自有的算法参数，不属于"识别规则"）
# --------------------------------------------------------------------------- #
INPUT_SR = 48000                 # 第一版唯一支持的采集采样率
DECIMATE = INPUT_SR // TARGET_SR  # 3
CUTOFF_HZ = 8000.0               # 抗混叠低通截止 = 目标奈奎斯特
FIR_TAPS = 201                   # 实测：201 抽头偏差 0.0008，101 抽头 0.0125
FIR_BETA = 8.0                   # Kaiser 窗 beta，实测取值

# 群延迟（只用于文档与测试，不参与补偿）
GROUP_DELAY_SAMPLES = (FIR_TAPS - 1) // 2            # 100 个 48 kHz 样本
GROUP_DELAY_MS = GROUP_DELAY_SAMPLES / INPUT_SR * 1000.0

# 预热帧数：每次会话开始时真实丢弃的窗口数（方案甲）
WARMUP_DROP_FRAMES = 2

__all__ = [
    "UnsupportedSampleRate",
    "StreamResampler48k",
    "WindowBuffer",
    "WarmupGate",
    "StreamingRecognizer",
    "SessionStats",
    "WINDOW",
    "HOP",
    "WINDOW_SECONDS",
    "WINDOW_SAMPLES",
    "HOP_SAMPLES",
    "TARGET_SR",
    "INPUT_SR",
    "DECIMATE",
    "FIR_TAPS",
    "GROUP_DELAY_MS",
    "WARMUP_DROP_FRAMES",
    "THETA_DEFAULT",
    "SMOOTH_N",
    "GROUPS",
]


class UnsupportedSampleRate(ValueError):
    """采集采样率不在支持列表内。第一版只支持 48000，明确报错，不静默降级。"""


# --------------------------------------------------------------------------- #
# FIR 设计
# --------------------------------------------------------------------------- #
def _kaiser_window(taps: int, beta: float) -> np.ndarray:
    """Kaiser 窗（numpy 自带 i0，不需要 scipy）。"""
    n = np.arange(taps, dtype=np.float64)
    r = 2.0 * n / (taps - 1) - 1.0
    return np.i0(beta * np.sqrt(np.clip(1.0 - r * r, 0.0, None))) / np.i0(beta)


def design_lowpass(taps: int = FIR_TAPS, sample_rate: int = INPUT_SR,
                   cutoff_hz: float = CUTOFF_HZ, beta: float = FIR_BETA) -> np.ndarray:
    """线性相位低通 FIR：窗函数法 sinc，截至 cutoff_hz，直流增益归一化为 1。"""
    if taps < 3 or taps % 2 == 0:
        raise ValueError("taps 必须是 >=3 的奇数，才能有整数群延迟")
    cutoff = cutoff_hz / float(sample_rate)      # 归一化到 周期/样本
    n = np.arange(taps, dtype=np.float64)
    mid = (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * (n - mid)) * _kaiser_window(taps, beta)
    total = h.sum()
    if total == 0.0:
        raise ValueError("FIR 设计失败（系数和为 0）")
    return h / total


# --------------------------------------------------------------------------- #
# 重采样：48 kHz -> 16 kHz
# --------------------------------------------------------------------------- #
class StreamResampler48k:
    """把 48000 Hz 的连续流降采样到 16000 Hz（因果 FIR + 3:1 抽取）。

    状态只有两个：`_hist`（前 taps-1 个输入样本）与 `in_pos`（已吃进的输入样本
    总数）。抽取按**全局** in_pos 定相位，所以 push 的块大小不影响输出。
    """

    def __init__(self, sample_rate: int = INPUT_SR, taps: int = FIR_TAPS):
        sample_rate = int(sample_rate)
        if sample_rate != INPUT_SR:
            raise UnsupportedSampleRate(
                f"第一版只支持 {INPUT_SR} Hz 采集，收到 {sample_rate} Hz。"
                f"不做自动降级：请改用支持 {INPUT_SR} Hz 的输入设备或宿主 API。"
            )
        self.sample_rate = sample_rate
        self.taps = int(taps)
        self.coefficients = design_lowpass(self.taps, sample_rate)
        self.reset()

    # -- 状态 ---------------------------------------------------------------
    def reset(self) -> None:
        self._hist = np.zeros(self.taps - 1, dtype=np.float64)
        self.in_pos = 0            # 已吃进的输入样本总数
        self.out_pos = 0           # 已产出的 16 kHz 样本总数

    # -- 主入口 -------------------------------------------------------------
    def push(self, block) -> np.ndarray:
        """喂一块输入样本，返回本次能确定产出的 16 kHz 样本（float32）。"""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        xp = np.concatenate([self._hist, x.astype(np.float64)])
        y = np.convolve(xp, self.coefficients, mode="valid")
        assert y.size == x.size, "因果卷积输出长度应等于本次输入长度"

        first = self.in_pos
        self._hist = xp[-(self.taps - 1):]
        self.in_pos += x.size

        idx = np.arange(first, first + y.size, dtype=np.int64)
        out = y[idx % DECIMATE == 0].astype(np.float32)
        self.out_pos += out.size
        return out

    # -- 便捷 ---------------------------------------------------------------
    def latency_ms(self) -> float:
        return GROUP_DELAY_MS


# --------------------------------------------------------------------------- #
# 定长切窗
# --------------------------------------------------------------------------- #
class WindowBuffer:
    """把 16 kHz 连续流切成窗口 k = [k*HOP, k*HOP + WINDOW)。

    用 `buf_start` + `stream_pos` 维护绝对时间轴：缓冲会被清理，但两个计数只增
    不减，所以任何时刻都能算出窗口起点，与清理次数无关。

    稳态缓冲长度上界 = WINDOW + 单次 push 的块长（清理后通常 < HOP + 块长）。
    """

    def __init__(self, window: int = WINDOW, hop: int = HOP):
        if not (0 < hop <= window):
            raise ValueError("需要 0 < hop <= window")
        self.window = int(window)
        self.hop = int(hop)
        self.reset()

    # -- 状态 ---------------------------------------------------------------
    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self.buf_start = 0        # _buf[0] 对应的全局 16 kHz 样本序号
        self.stream_pos = 0       # 已收到的全局 16 kHz 样本总数
        self.next_frame = 0       # 下一个待产出的窗口索引 k
        self.trim_count = 0       # 清理次数（仅诊断用）
        self.padded_samples = 0   # flush() 补零样本数（仅离线）

    # -- 只读属性 -----------------------------------------------------------
    @property
    def buffered(self) -> int:
        """当前缓冲里保留的样本数（= stream_pos - buf_start）。"""
        return int(self.stream_pos - self.buf_start)

    def frames_ready(self) -> int:
        """按当前 stream_pos 还能产出多少帧（不补零）。"""
        if self.stream_pos < self.window:
            return 0
        return (self.stream_pos - self.window) // self.hop + 1 - self.next_frame

    # -- 主入口 -------------------------------------------------------------
    def push(self, block) -> List[Tuple[int, np.ndarray]]:
        """喂一块 16 kHz 样本，返回本次新凑满的 [(k, window), ...]。"""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if x.size:
            if self._buf.size:
                self._buf = np.concatenate([self._buf, x])
            else:
                self._buf = x.copy()      # 避免持有调用方的大数组视图
            self.stream_pos += x.size
        return self._emit()

    def _emit(self) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        while self.next_frame * self.hop + self.window <= self.stream_pos:
            k = self.next_frame
            off = k * self.hop - self.buf_start
            out.append((k, self._buf[off:off + self.window].copy()))
            self.next_frame += 1

            # 下一帧起点之前的数据再也用不到，立即丢弃（绝对计数不变）
            drop = self.next_frame * self.hop - self.buf_start
            if drop > 0:
                self._buf = self._buf[drop:]
                self.buf_start += drop
                self.trim_count += 1
        return out

    def flush(self) -> List[Tuple[int, np.ndarray]]:
        """离线专用：把尾部不足一个窗口的帧用零补齐后一并产出。

        语义与 yamnet_recognizer.iter_windows(pad_tail=True) 完全一致：
        帧 k 的个数 = ceil(总样本 / HOP)，最后一帧补零。
        实时链路**不要调用**（真实麦克风不会结束）。
        本方法不修改 stream_pos / buf_start，补零量单独记在 padded_samples。
        """
        out: List[Tuple[int, np.ndarray]] = []
        if self.stream_pos == 0:
            return out
        last_k = (self.stream_pos + self.hop - 1) // self.hop - 1
        pad_total = 0
        while self.next_frame <= last_k:
            k = self.next_frame
            off = k * self.hop - self.buf_start
            seg = self._buf[off:off + self.window]
            if seg.size < self.window:
                pad = self.window - seg.size
                seg = np.concatenate([seg, np.zeros(pad, dtype=np.float32)])
                pad_total += pad
            out.append((k, seg.copy()))
            self.next_frame += 1
        self.padded_samples += pad_total
        return out


# --------------------------------------------------------------------------- #
# 预热门
# --------------------------------------------------------------------------- #
class WarmupGate:
    """会话开始时丢弃前 N 个窗口的识别结果。

    "只丢一次"由单调计数器保证：计数到 N 之后永久放行。缓冲区清理、后续分块、
    长时间运行都不会让它重新计数；只有 reset()（用户重新开始监听）才归零。
    """

    def __init__(self, drop_frames: int = WARMUP_DROP_FRAMES):
        self.drop_frames = int(drop_frames)
        if self.drop_frames < 0:
            raise ValueError("drop_frames 不能为负")
        self.reset()

    def reset(self) -> None:
        self.dropped = 0
        self.seen = 0

    def accept(self) -> bool:
        """True = 该窗口参与判定；False = 该窗口属于预热，丢弃。"""
        self.seen += 1
        if self.dropped < self.drop_frames:
            self.dropped += 1
            return False
        return True


# --------------------------------------------------------------------------- #
# 会话统计
# --------------------------------------------------------------------------- #
@dataclass
class SessionStats:
    """一次监听会话的可上报状态（阶段 2 的 /api/listen/status 直接映射它）。"""

    input_rate: int = INPUT_SR
    stream_pos: int = 0                 # 已收到 16 kHz 样本总数
    buf_start: int = 0                  # 缓冲中首个样本的全局序号
    buffered: int = 0                   # 当前缓冲样本数
    frames_produced: int = 0            # 已产出窗口数
    frames_scored: int = 0              # 实际送进模型判定的窗口数
    warmup_frames_dropped: int = 0      # 预热丢弃数（会话内累计，恒 <= 2）
    triggers: int = 0
    trim_count: int = 0                 # 缓冲清理次数（诊断）

    def as_dict(self) -> Dict[str, object]:
        return {
            "input_rate": self.input_rate,
            "stream_pos": self.stream_pos,
            "buf_start": self.buf_start,
            "buffered": self.buffered,
            "frames_produced": self.frames_produced,
            "frames_scored": self.frames_scored,
            "warmup_frames_dropped": self.warmup_frames_dropped,
            "triggers": self.triggers,
            "trim_count": self.trim_count,
        }


# --------------------------------------------------------------------------- #
# 识别会话
# --------------------------------------------------------------------------- #
class StreamingRecognizer:
    """一条完整的流式识别链路：块 -> (重采样) -> 切窗 -> 预热 -> 3 帧判定。

    参数
    ----
    input_rate    : 48000 = 麦克风原始流（走本模块 FIR）；
                    16000 = 波形已经是 16 kHz（层 A 离线回归 / 上游已重采样）。
                    其它值直接抛 UnsupportedSampleRate，不降级。
    theta         : 默认 THETA_DEFAULT（实验值，未标定）
    smooth_n      : 默认 SMOOTH_N
    cooldown_s    : 同组重复触发冷却，默认 0（与离线口径一致）
    warmup_frames : 默认 WARMUP_DROP_FRAMES（方案甲 = 2）
    """

    def __init__(
        self,
        bundle: ModelBundle,
        input_rate: int = INPUT_SR,
        theta: float = THETA_DEFAULT,
        smooth_n: int = SMOOTH_N,
        cooldown_s: float = 0.0,
        warmup_frames: int = WARMUP_DROP_FRAMES,
    ):
        input_rate = int(input_rate)
        if input_rate == TARGET_SR:
            self.resampler: Optional[StreamResampler48k] = None
        elif input_rate == INPUT_SR:
            self.resampler = StreamResampler48k(input_rate)
        else:
            raise UnsupportedSampleRate(
                f"输入采样率 {input_rate} Hz 不支持：只接受 {INPUT_SR}（麦克风）"
                f"或 {TARGET_SR}（已重采样的波形）。不自动降级。"
            )

        self.bundle = bundle
        self.input_rate = input_rate
        self.buffer = WindowBuffer()
        self.warmup = WarmupGate(warmup_frames)
        self.judge = FrameJudge(theta=theta, smooth_n=smooth_n, cooldown_s=cooldown_s)
        self.stats = SessionStats(input_rate=input_rate)
        self.records: List[FrameRecord] = []
        self.triggers: List[Trigger] = []
        self.warmup_end_time: Optional[float] = None   # 预热最后一帧的判定时刻

    # -- 会话生命周期 -------------------------------------------------------
    def reset_session(self) -> None:
        """stop -> start：清空本次会话的所有状态。

        清零：重采样器状态、窗口缓冲、全局时间轴、预热计数、3 帧滑窗、冷却、
        计数器、本次会话的事件列表。
        不清：模型（进程内单例）、数据库历史事件（本模块根本不碰数据库）。
        """
        if self.resampler is not None:
            self.resampler.reset()
        self.buffer.reset()
        self.warmup.reset()
        self.judge.reset()
        self.stats = SessionStats(input_rate=self.input_rate)
        self.records.clear()
        self.triggers.clear()
        self.warmup_end_time = None

    # -- 主入口 -------------------------------------------------------------
    def push(self, block) -> List[FrameRecord]:
        """喂一块采集数据，返回本次新产生的判定记录（预热帧不返回）。"""
        if self.resampler is not None:
            samples = self.resampler.push(block)
        else:
            samples = np.asarray(block, dtype=np.float32).reshape(-1)
        return self._consume(samples)

    def flush(self) -> List[FrameRecord]:
        """离线专用尾部补零（实时链路不调用）。"""
        return self._consume(None, flush=True)

    def _consume(self, samples, flush: bool = False) -> List[FrameRecord]:
        windows = self.buffer.flush() if flush else self.buffer.push(samples)
        out: List[FrameRecord] = []
        for k, window in windows:
            self.stats.frames_produced += 1
            if not self.warmup.accept():
                self.stats.warmup_frames_dropped += 1
                self.warmup_end_time = k * HOP_SECONDS + DECISION_LATENCY_S
                continue
            t = k * HOP_SECONDS + DECISION_LATENCY_S
            record = self.judge.push_window(self.bundle, window, t)
            # 判据来自 FrameRecord.frame_index（判定序号），时间来自窗口序号 k，
            # 两者在预热后不同，故把 k 一并挂在记录上，方便排障与对齐。
            setattr(record, "window_index", k)
            self.records.append(record)
            self.triggers.extend(record.triggers)
            self.stats.frames_scored += 1
            out.append(record)
        self.stats.stream_pos = self.buffer.stream_pos
        self.stats.buf_start = self.buffer.buf_start
        self.stats.buffered = self.buffer.buffered
        self.stats.trim_count = self.buffer.trim_count
        self.stats.triggers = len(self.triggers)
        return out

    # -- 便捷 ---------------------------------------------------------------
    def first_trigger(self) -> Optional[Trigger]:
        return self.triggers[0] if self.triggers else None

    def triggered_groups(self) -> List[str]:
        seen: List[str] = []
        for t in self.triggers:
            if t.group not in seen:
                seen.append(t.group)
        return seen

    def peak_smoothed(self) -> Dict[str, float]:
        """各组 3 帧均值的峰值（没有判定记录时为 0）。"""
        peaks = {g: 0.0 for g in GROUPS}
        for rec in self.records:
            for g, v in rec.smoothed.items():
                if v is not None and v > peaks.get(g, 0.0):
                    peaks[g] = float(v)
        return peaks


# --------------------------------------------------------------------------- #
# 一行式：整段波形 -> 结果（自测用；实时链路按块 push）
# --------------------------------------------------------------------------- #
def run_stream(
    bundle: ModelBundle,
    mono: np.ndarray,
    input_rate: int = INPUT_SR,
    chunk: int = 4096,
    warmup_frames: int = WARMUP_DROP_FRAMES,
    theta: float = THETA_DEFAULT,
) -> StreamingRecognizer:
    """按固定块长把整段波形喂完（含尾部补零），返回跑完的识别会话。"""
    rec = StreamingRecognizer(bundle, input_rate=input_rate, theta=theta,
                              warmup_frames=warmup_frames)
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    for i in range(0, mono.size, chunk):
        rec.push(mono[i:i + chunk])
    rec.flush()
    return rec


if __name__ == "__main__":
    # 不加载模型的最小自检：只验证重采样与切窗的数学性质
    ok = True
    print("=== yamnet_stream 最小自检（不加载模型）===")
    print(f"  WINDOW={WINDOW} ({WINDOW_SECONDS:.4f}s)  HOP={HOP}  DECIMATE={DECIMATE}")
    print(f"  FIR taps={FIR_TAPS} beta={FIR_BETA} 群延迟={GROUP_DELAY_SAMPLES} 样本"
          f"={GROUP_DELAY_MS:.3f} ms")

    h = design_lowpass()
    ok &= abs(h.sum() - 1.0) < 1e-12 and h.size == FIR_TAPS
    print(f"  低通系数: 长度 {h.size} 和 {h.sum():.12f}  {'OK' if ok else 'FAIL'}")

    rng = np.random.default_rng(0)
    x48 = (rng.standard_normal(INPUT_SR * 3) * 0.1).astype(np.float32)
    a = StreamResampler48k()
    whole = a.push(x48)
    b = StreamResampler48k()
    parts = [b.push(x48[i:i + 4096]) for i in range(0, x48.size, 4096)]
    piece = np.concatenate(parts)
    d = float(np.max(np.abs(whole - piece))) if whole.size else -1.0
    ok &= whole.size == piece.size and d < 1e-12
    print(f"  分块不变性: 整段 {whole.size} 样本 / 4096 块 {piece.size} 样本 / maxdiff {d:.3e}"
          f"  {'OK' if d < 1e-12 else 'FAIL'}")

    wb = WindowBuffer()
    frames = wb.push(np.zeros(WINDOW, dtype=np.float32))
    ok &= len(frames) == 1 and frames[0][0] == 0
    extra = wb.push(np.zeros(HOP, dtype=np.float32))
    ok &= len(extra) == 1 and extra[0][0] == 1
    print(f"  切窗: 首批 {len(frames)} 帧, 再补一个 HOP 得 {len(extra)} 帧  "
          f"{'OK' if len(extra) == 1 else 'FAIL'}")

    try:
        StreamResampler48k(44100)
        ok = False
        print("  44.1 kHz -> 未抛异常  FAIL")
    except UnsupportedSampleRate as exc:
        print(f"  44.1 kHz -> UnsupportedSampleRate  OK ({exc})")

    print()
    print("自检结果:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
