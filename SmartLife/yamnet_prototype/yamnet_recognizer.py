#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SmartLife YAMNet 识别库（纯计算，不含麦克风、不含 HTTP、不写数据库）。

本模块只做三件事：
    波形 -> 单帧分数 -> 分组分数 -> 触发判定

设计要点
--------
1. 模块导入时不加载 TensorFlow。`load_model()` 才真正 import TF 并读模型，
   因此 `import yamnet_recognizer` 的开销是毫秒级，适合被 Web 进程引用。
2. 规则只有一份：`GROUPS` / `SMOOTH_N` / `THETA_DEFAULT` / `HOP_S` 全部从
   `yamnet_doorbell_eval` 导入，本模块不复制类别清单，避免两处定义漂移。
3. 窗口长度经过实测标定（见下方 WINDOW_SAMPLES 说明），保证流式逐窗推理
   的分数与「整段离线推理」的对应帧一致（实测最大偏差 6e-8，即浮点误差级）。

窗口长度的由来（重要，不要随意改）
----------------------------------
YAMNet 的一帧需要 **96 个 STFT 帧**（frame_length=400, frame_step=160 @16 kHz）。
STFT 帧数 = 1 + floor((N - 400) / 160)。

    N = 15360 (0.96 s, 原离线窗) -> 94 帧  -> 不足 96，模型内部补零
    N = 15600 (0.975 s)          -> 96 帧  -> 正好，无补零  <<< 本模块采用
    N = 15680 (0.98 s)           -> 96 帧  -> 正好，但会多输出 1 帧（浪费）

补零会污染这一帧的分数：实测 0.96 s 单窗与离线对应帧的偏差可达 **0.20**，
足以改变触发结论；15,600 样本则把偏差压到 **6e-8**。

因此：
    WINDOW_SAMPLES = 15600  (0.975 s)
    HOP_SAMPLES    = 7680   (0.480 s)
    帧 k 的窗口 = 波形[k*7680 : k*7680 + 15600]

代价：比原 0.96 s 窗多 15 ms 延迟，可忽略。
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 路径与规则来源（单一真源）
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# GROUPS / SMOOTH_N / THETA_DEFAULT / WIN_S / HOP_S 一律从既有评估脚本导入。
# 这是本模块唯一的规则来源：改规则只改 yamnet_doorbell_eval.py 一处。
from yamnet_doorbell_eval import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    GROUPS,
    HOP_S,
    SMOOTH_N,
    THETA_DEFAULT,
    WIN_S,
)

# --------------------------------------------------------------------------- #
# 音频常量
# --------------------------------------------------------------------------- #
TARGET_SR = 16000                 # YAMNet 固定输入采样率
STFT_FRAME_LENGTH = 400           # YAMNet 内部 STFT 窗长（样本）
STFT_FRAME_STEP = 160             # YAMNet 内部 STFT 跳距（样本）
PATCH_FRAMES = 96                 # 一帧 patch 需要的 STFT 帧数

# 恰好凑满 96 个 STFT 帧的最小样本数：400 + (96 - 1) * 160 = 15600
WINDOW_SAMPLES = STFT_FRAME_LENGTH + (PATCH_FRAMES - 1) * STFT_FRAME_STEP  # 15600
HOP_SAMPLES = int(round(HOP_S * TARGET_SR))                                # 7680

WINDOW_S = WINDOW_SAMPLES / TARGET_SR   # 0.975 s（离线口径记为 0.96 s，差 15 ms）
HOP_SECONDS = HOP_SAMPLES / TARGET_SR   # 0.480 s

# 触发时刻的计时原点：与 yamnet_doorbell_eval 的 first_trigger_time 保持同一套
# 语义（"判定完成的那一刻"），但用真实窗长而非离线文档里的 0.96。
DECISION_LATENCY_S = WINDOW_S

assert WINDOW_SAMPLES == 15600, "窗口长度被改动，请先重新验证与离线帧的等价性"
assert HOP_SAMPLES == 7680


# --------------------------------------------------------------------------- #
# 模型加载（懒加载 + 进程内单例）
# --------------------------------------------------------------------------- #
class ModelLoadError(RuntimeError):
    """模型加载失败：目录缺失、TensorFlow 导入失败、模型文件损坏等。"""


class _ModelBundle:
    """模型 + 类别表的只读组合，加载一次后复用。"""

    __slots__ = ("model", "class_names", "name_to_idx", "group_idx", "model_dir")

    def __init__(self, model, class_names, name_to_idx, group_idx, model_dir):
        self.model = model
        self.class_names = class_names
        self.name_to_idx = name_to_idx
        self.group_idx = group_idx
        self.model_dir = model_dir


_CACHE: Dict[str, _ModelBundle] = {}


def load_model(model_dir: Optional[str] = None, verbose: bool = False) -> _ModelBundle:
    """加载 YAMNet（进程内只加载一次，重复调用直接返回缓存）。

    参数
    ----
    model_dir : 模型目录，默认 <本文件目录>/models/yamnet
    verbose   : 是否打印加载耗时

    异常
    ----
    ModelLoadError : 目录不存在 / 缺少必需文件 / TF 导入或加载失败
    """
    model_dir = os.path.abspath(model_dir or DEFAULT_MODEL_DIR)
    cached = _CACHE.get(model_dir)
    if cached is not None:
        return cached

    if not os.path.isdir(model_dir):
        raise ModelLoadError(f"模型目录不存在: {model_dir}")

    required = ("saved_model.pb", "variables", "assets")
    missing = [n for n in required if not os.path.exists(os.path.join(model_dir, n))]
    if missing:
        raise ModelLoadError(f"模型目录不完整，缺少: {', '.join(missing)} ({model_dir})")

    try:
        import tensorflow_hub as hub  # 延迟 import：模块级不拉 TF
    except Exception as exc:  # pragma: no cover - 环境问题
        raise ModelLoadError(f"无法导入 tensorflow_hub: {exc}") from exc

    try:
        import time

        t0 = time.perf_counter()
        # 强制 CPU、关掉冗余日志，避免被 Web 进程的 stdout 污染
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        model = hub.load(model_dir)
        elapsed = time.perf_counter() - t0
    except Exception as exc:  # pragma: no cover - 环境问题
        raise ModelLoadError(f"加载模型失败: {exc}") from exc

    class_map = model.class_map_path().numpy().decode("utf-8")
    from yamnet_batch_eval import load_class_names

    class_names = load_class_names(class_map)
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    missing_cls = [c for group in GROUPS.values() for c in group if c not in name_to_idx]
    if missing_cls:
        raise ModelLoadError(f"类别表中找不到这些类: {missing_cls}")

    group_idx = {g: [name_to_idx[c] for c in classes] for g, classes in GROUPS.items()}

    bundle = _ModelBundle(model, class_names, name_to_idx, group_idx, model_dir)
    _CACHE[model_dir] = bundle

    if verbose:
        print(f"[recognizer] 模型已加载: {model_dir}  ({elapsed:.2f} s, {len(class_names)} 类)")
    return bundle


def is_loaded(model_dir: Optional[str] = None) -> bool:
    """模型是否已在当前进程内加载（不触发加载）。"""
    return os.path.abspath(model_dir or DEFAULT_MODEL_DIR) in _CACHE


def unload_model(model_dir: Optional[str] = None) -> None:
    """从缓存移除（主要给测试用，正常服务生命周期内不需要）。"""
    _CACHE.pop(os.path.abspath(model_dir or DEFAULT_MODEL_DIR), None)


# --------------------------------------------------------------------------- #
# 单窗打分
# --------------------------------------------------------------------------- #
def score_window(bundle: _ModelBundle, waveform: Sequence[float]) -> np.ndarray:
    """对一个窗口（应为 WINDOW_SAMPLES 长）推理，返回 (521,) 的分数向量。

    输入不足 WINDOW_SAMPLES 时补零；超过时截取前 WINDOW_SAMPLES（调用方应自行切片）。
    返回的是多标签 sigmoid 输出，**不是概率，也不是准确率**。
    """
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wav.size < WINDOW_SAMPLES:
        wav = np.pad(wav, (0, WINDOW_SAMPLES - wav.size))
    elif wav.size > WINDOW_SAMPLES:
        wav = wav[:WINDOW_SAMPLES]

    scores, _embeddings, _log_mel = bundle.model(wav)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None, :]
    return scores[0]   # 15600 样本恰好输出 1 帧


def group_scores(bundle: _ModelBundle, frame_scores: np.ndarray) -> Dict[str, float]:
    """单帧分数向量 -> {组名: 组内最大类分数}。"""
    return {
        g: float(frame_scores[cols].max())
        for g, cols in bundle.group_idx.items()
    }


# --------------------------------------------------------------------------- #
# 窗口切分（供自测与实时管线共用）
# --------------------------------------------------------------------------- #
def iter_windows(
    mono: np.ndarray,
    window: int = WINDOW_SAMPLES,
    hop: int = HOP_SAMPLES,
    pad_tail: bool = True,
) -> Iterator[Tuple[int, np.ndarray]]:
    """按实时流的方式切窗：yield (帧索引 k, 长 window 的片段)。

    帧 k 的窗口是 mono[k*hop : k*hop + window]。

    pad_tail=True 时，文件尾部不足一个窗口的部分会用零补齐，使最后一帧也能算出。
    这对应实时场景里"事件之后环境安静下来"的情形；补出来的帧分数天然很低，
    只会拉低判定量，不会凭空造出触发。pad_tail=False 则尾部帧直接丢弃。
    """
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    k = 0
    while True:
        start = k * hop
        if start >= mono.size:
            break
        seg = mono[start:start + window]
        if seg.size < window:
            if not pad_tail:
                break
            seg = np.pad(seg, (0, window - seg.size))
        yield k, seg
        k += 1


# --------------------------------------------------------------------------- #
# 触发判定
# --------------------------------------------------------------------------- #
@dataclass
class Trigger:
    """一次触发。score 是 3 帧滑动均值（未标定的模型分数，不是概率）。"""

    group: str                 # "doorbell" / "knock"
    score: float               # 3 帧均值
    frame_index: int           # 窗口最后一帧的索引
    time_s: float              # 判定完成时刻（秒，相对流起点）
    raw_frame_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class FrameRecord:
    """一帧的判定结果，便于日志与调试。"""

    frame_index: int
    time_s: float
    per_group: Dict[str, float]      # 各组原始帧分数
    smoothed: Dict[str, float]       # 各组 3 帧均值（不足 3 帧时为 None）
    triggers: List[Trigger]


class FrameJudge:
    """3 帧滑动均值 + 阈值 + 冷却 + 组间互斥。

    与离线评估完全同口径：
        smoothed[g] = mean(last N 帧的 group_max[g])
        触发条件    = smoothed[g] > theta

    额外只加两件工程设施（离线脚本里没有）：
        1. 冷却：同组触发后 cooldown_s 内不再重复触发；
        2. 互斥：同一帧两组同时过线时，只记分数更高的那一组。

    注意：阈值 theta 是实验值，未标定。0.45 只在这 12 段样本上验证过触发情况，
    不代表误报率或准确率。
    """

    def __init__(
        self,
        theta: float = THETA_DEFAULT,
        smooth_n: int = SMOOTH_N,
        cooldown_s: float = 0.0,
        groups: Optional[Sequence[str]] = None,
    ):
        self.theta = float(theta)
        self.smooth_n = int(smooth_n)
        self.cooldown_s = float(cooldown_s)
        self.groups = list(groups) if groups else list(GROUPS.keys())
        self._history: Dict[str, deque] = {g: deque(maxlen=self.smooth_n) for g in self.groups}
        self._last_trigger_t: Dict[str, float] = {}
        self.frames_seen = 0
        self.triggers: List[Trigger] = []

    # -- 状态 ---------------------------------------------------------------
    def reset(self) -> None:
        """清空滑窗与冷却状态。模型不卸载。"""
        for g in self.groups:
            self._history[g].clear()
        self._last_trigger_t.clear()
        self.frames_seen = 0
        self.triggers.clear()

    def set_theta(self, theta: float) -> None:
        self.theta = float(theta)

    def warm(self) -> bool:
        """是否已攒够 smooth_n 帧（可以开始判定）。"""
        return all(len(self._history[g]) >= self.smooth_n for g in self.groups)

    # -- 主入口 -------------------------------------------------------------
    def push(self, per_group: Dict[str, float], time_s: float) -> FrameRecord:
        """喂入一帧的分组分数，返回本帧判定结果。

        per_group : {"doorbell": 0.12, "knock": 0.03}
        time_s    : 该帧判定完成时刻（秒）
        """
        frame_index = self.frames_seen
        self.frames_seen += 1

        for g in self.groups:
            self._history[g].append(float(per_group.get(g, 0.0)))

        smoothed: Dict[str, float] = {}
        ready = True
        for g in self.groups:
            if len(self._history[g]) < self.smooth_n:
                ready = False
                smoothed[g] = float("nan")
            else:
                smoothed[g] = float(np.mean(self._history[g]))

        triggers: List[Trigger] = []
        if ready:
            # 先找所有过线的，再按分数取最高（组间互斥）
            over = [(g, smoothed[g]) for g in self.groups if smoothed[g] > self.theta]
            if over:
                over.sort(key=lambda x: x[1], reverse=True)
                g, s = over[0]
                last = self._last_trigger_t.get(g)
                cooling = last is not None and (time_s - last) < self.cooldown_s
                if not cooling:
                    trg = Trigger(
                        group=g,
                        score=round(s, 4),
                        frame_index=frame_index,
                        time_s=round(time_s, 3),
                        raw_frame_scores={k: round(float(v), 4) for k, v in per_group.items()},
                    )
                    triggers.append(trg)
                    self.triggers.append(trg)
                    self._last_trigger_t[g] = time_s

        return FrameRecord(
            frame_index=frame_index,
            time_s=round(time_s, 3),
            per_group={k: round(float(v), 6) for k, v in per_group.items()},
            smoothed={k: (round(v, 6) if v == v else None) for k, v in smoothed.items()},
            triggers=triggers,
        )

    # -- 便捷：直接喂一个窗口 ------------------------------------------------
    def push_window(self, bundle: _ModelBundle, waveform: Sequence[float], time_s: float) -> FrameRecord:
        scores = score_window(bundle, waveform)
        return self.push(group_scores(bundle, scores), time_s)


# --------------------------------------------------------------------------- #
# 一行式：整段波形 -> 所有触发（自测与离线复核用）
# --------------------------------------------------------------------------- #
def run_waveform(
    bundle: _ModelBundle,
    mono: np.ndarray,
    theta: float = THETA_DEFAULT,
    cooldown_s: float = 0.0,
    pad_tail: bool = True,
) -> Tuple[List[FrameRecord], List[Trigger]]:
    """把一段 16 kHz 单声道波形按实时流方式跑完，返回 (逐帧记录, 触发列表)。"""
    judge = FrameJudge(theta=theta, cooldown_s=cooldown_s)
    records: List[FrameRecord] = []
    for k, seg in iter_windows(mono, pad_tail=pad_tail):
        t = k * HOP_SECONDS + DECISION_LATENCY_S
        records.append(judge.push_window(bundle, seg, t))
    return records, judge.triggers


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """最小自检：不加载模型，只验证常量与判据逻辑。"""
    ok = True
    print("=== yamnet_recognizer 自检（不加载模型）===")
    print(f"  WINDOW_SAMPLES = {WINDOW_SAMPLES}  ({WINDOW_S:.4f} s)")
    print(f"  HOP_SAMPLES    = {HOP_SAMPLES}  ({HOP_SECONDS:.4f} s)")
    print(f"  theta          = {THETA_DEFAULT}  smooth_n = {SMOOTH_N}")
    print(f"  组              = {GROUPS}")
    ok &= WINDOW_SAMPLES == 15600
    ok &= HOP_SAMPLES == 7680
    ok &= set(GROUPS) == {"doorbell", "knock"}

    # 判据逻辑：3 帧均值 + 冷却 + 互斥
    j = FrameJudge(theta=0.45, cooldown_s=10.0)
    r1 = j.push({"doorbell": 0.10, "knock": 0.00}, 1.0)
    r2 = j.push({"doorbell": 0.60, "knock": 0.00}, 1.5)
    r3 = j.push({"doorbell": 0.90, "knock": 0.00}, 2.0)   # 3 帧齐 -> 均值 0.533 > 0.45
    ok &= (r1.triggers == [] and r2.triggers == [])
    ok &= len(r3.triggers) == 1 and r3.triggers[0].group == "doorbell"
    print(f"  攒满 3 帧后均值 {r3.smoothed['doorbell']:.3f} -> 触发 {len(r3.triggers)} 次  "
          f"{'OK' if r3.triggers else 'FAIL'}")

    r4 = j.push({"doorbell": 0.90, "knock": 0.10}, 3.0)    # 冷却内，不应再触发
    ok &= r4.triggers == []
    print(f"  冷却期内重复 -> 触发 {len(r4.triggers)} 次  {'OK' if not r4.triggers else 'FAIL'}")

    # 互斥：两组都过线，取分高的
    j2 = FrameJudge(theta=0.45, cooldown_s=0.0)
    for _ in range(3):
        rj = j2.push({"doorbell": 0.50, "knock": 0.80}, 5.0)
    ok &= len(rj.triggers) == 1 and rj.triggers[0].group == "knock"
    print(f"  两组同时过线 -> 取 {rj.triggers[0].group if rj.triggers else '无'}  "
          f"{'OK' if rj.triggers and rj.triggers[0].group == 'knock' else 'FAIL'}")

    j3 = FrameJudge(theta=0.45)
    j3.push({"doorbell": 0.10, "knock": 0.10}, 1.0)
    j3.push({"doorbell": 0.10, "knock": 0.10}, 1.5)
    rw = j3.push({"doorbell": 0.10, "knock": 0.10}, 2.0)
    ok &= rw.triggers == []
    print(f"  全低分不触发  {'OK' if not rw.triggers else 'FAIL'}")

    # 窗口切分：帧数 = ceil(总样本 / 跳距)
    fake = np.zeros(WINDOW_SAMPLES + 2 * HOP_SAMPLES, dtype=np.float32)
    wins = list(iter_windows(fake))
    expect = int(np.ceil(fake.size / HOP_SAMPLES))   # 30960 / 7680 -> 5
    ok &= len(wins) == expect and all(w.size == WINDOW_SAMPLES for _, w in wins)
    print(f"  切窗: {len(wins)} 个（期望 {expect}），每个 {WINDOW_SAMPLES} 样本  "
          f"{'OK' if len(wins) == expect else 'FAIL'}")

    # 尾部丢弃模式
    wins_np = list(iter_windows(fake, pad_tail=False))
    expect_np = (fake.size - WINDOW_SAMPLES) // HOP_SAMPLES + 1
    ok &= len(wins_np) == expect_np
    print(f"  切窗(pad_tail=False): {len(wins_np)} 个（期望 {expect_np}）  "
          f"{'OK' if len(wins_np) == expect_np else 'FAIL'}")

    print()
    print("自检结果:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
