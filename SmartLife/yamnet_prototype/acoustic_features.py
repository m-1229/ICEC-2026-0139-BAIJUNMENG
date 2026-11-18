#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SmartLife 声学特征影子检测。

本模块只做流式 DSP 特征提取，不加载 YAMNet、不访问麦克风、不写文件，也不
产生正式提醒。它复用上层已有的 48 kHz / 单声道 / float32 / 20 ms 音频块，
输出“门铃候选”和“敲门候选”，供监听状态接口做第一阶段影子观察。

重要边界
--------
1. 不做音频增益归一化，不修改调用方传入的音频。
2. 所有电平判断都相对于自适应噪声底，而不是写死某个麦克风增益。
3. 候选是规则证据，不是校准概率，不能直接等同于真实事件。
4. 检测器只依赖 numpy；主项目 pyproject/requirements 不新增依赖。

两个分支
--------
门铃
    持续音调：稳定的低频谱平坦度、相邻频谱形状相似度、合适的频谱重心，
    并在超过最短持续时间后产生候选。过短音调会进入重复音调节奏缓存，
    在窗口内重复出现后才产生候选。

敲门
    先检测能量快速上升的起音，再等待短时包络确认是否快速衰减。确认后的
    冲击按时间间隔聚合，至少两次冲击才形成敲门候选；孤立冲击不会成为
    高可信敲门候选。

状态重置
--------
`reset()` 清空全部缓冲区、噪声底、音调段、待确认冲击和候选历史。麦克风
队列丢失音频后，上层应调用它，避免把缺口两侧拼成同一个节奏。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal, Optional, Sequence

import numpy as np


SAMPLE_RATE = 48000
FRAME_SAMPLES = 960                 # 20 ms
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE

# 噪声底校准与跟随
NOISE_CALIBRATION_FRAMES = 10       # 200 ms
NOISE_FALL_ALPHA = 0.25
NOISE_RISE_ALPHA = 0.02
NOISE_ACTIVE_RISE_ALPHA = 0.01
NOISE_RISE_LIMIT_DB = 6.0
NOISE_ACTIVE_ADAPT_AFTER_FRAMES = 100   # 连续 2 秒活动后才按背景噪声跟随

# 活跃门限
ACTIVITY_MARGIN_DB = 10.0
ONSET_MARGIN_DB = 12.0

# 门铃音调规则
TONE_MIN_DURATION_S = 0.14
TONE_MAX_GAP_FRAMES = 2             # 最多容忍 40 ms 掉帧
TONE_CONFIRM_FRAMES = 2
TONE_FLATNESS_MAX = 0.12
TONE_SIMILARITY_MIN = 0.985
TONE_MIN_CENTROID_HZ = 500.0
TONE_MAX_CENTROID_HZ = 4500.0
REPEATED_TONE_MIN_COUNT = 2
REPEATED_TONE_MIN_PULSE_DURATION_S = 0.05
REPEATED_TONE_WINDOW_S = 2.5
DOORBELL_EMIT_COOLDOWN_S = 1.5

# 敲门冲击规则
IMPACT_MIN_RISE_DB = 8.0
IMPACT_CONFIRM_FRAMES = 2
IMPACT_MIN_DURATION_S = 0.02
IMPACT_MAX_DURATION_S = 0.35
IMPACT_DECAY_DB = 6.0
IMPACT_MIN_CENTROID_HZ = 100.0
IMPACT_MAX_CENTROID_HZ = 2500.0
KNOCK_MIN_COUNT = 2
KNOCK_MAX_COUNT = 8
KNOCK_MIN_INTERVAL_S = 0.08
KNOCK_MAX_INTERVAL_S = 1.50
KNOCK_WINDOW_S = 3.0
KNOCK_EMIT_COOLDOWN_S = 1.5
ONSET_REFRACTORY_S = 0.12

# 相邻频谱与起音
SPECTRUM_MIN_HZ = 100.0
SPECTRUM_MAX_HZ = 7000.0
ONSET_SPECTRAL_FLUX_MIN = 0.08

CandidateType = Literal["doorbell", "knock"]


@dataclass(frozen=True)
class AcousticFrame:
    """一帧 20 ms 音频的特征快照（诊断用）。"""

    frame_index: int
    start_time_s: float
    end_time_s: float
    rms_dbfs: float
    noise_floor_dbfs: float
    relative_db: float
    crest_factor: float
    spectral_flatness: float
    spectral_centroid_hz: float
    dominant_hz: float
    spectral_similarity: Optional[float]
    spectral_flux: float
    energy_rise_db: float
    active: bool
    tonal: bool


@dataclass(frozen=True)
class AcousticCandidate:
    """一条影子候选，不进入正式提醒，也不写数据库。"""

    candidate_type: CandidateType
    start_time_s: float
    time_s: float
    duration_s: float
    relative_db: float
    reason: str
    features: Dict[str, float | int | str]


@dataclass
class _ToneSegment:
    start_frame: int
    last_frame: int
    peak_relative_db: float
    centroid_sum: float
    similarity_sum: float
    similarity_count: int
    flatness_sum: float
    frame_count: int
    emitted: bool = False


@dataclass
class _TonePulse:
    start_time_s: float
    duration_s: float
    centroid_hz: float
    similarity: float
    relative_db: float


@dataclass
class _PendingImpact:
    start_frame: int
    onset_relative_db: float
    rise_db: float
    spectral_flux: float
    peak_db: float
    peak_frame: int
    last_db: float
    centroid_sum: float
    similarity_sum: float
    similarity_count: int
    similarity_max: float
    flatness_sum: float
    frame_count: int


@dataclass
class _ConfirmedImpact:
    start_time_s: float
    time_s: float
    duration_s: float
    relative_db: float
    centroid_hz: float
    similarity: Optional[float]
    flatness: float
    decay_db: float


@dataclass
class _KnockCluster:
    cluster_id: str
    impacts: List[_ConfirmedImpact] = field(default_factory=list)


class AcousticShadowDetector:
    """将任意大小的 48 kHz 音频块转换为影子候选。

    `push()` 接受任意长度的一维块，也接受形状为 ``(n, 1)`` 的 sounddevice
    数据。输出只取决于音频样本，不取决于每次 push 的块边界。
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        if int(sample_rate) != SAMPLE_RATE:
            raise ValueError(f"声学影子检测只支持 {SAMPLE_RATE} Hz")
        self.sample_rate = int(sample_rate)
        self._window = np.hanning(FRAME_SAMPLES).astype(np.float64)
        self._freqs = np.fft.rfftfreq(FRAME_SAMPLES, 1.0 / SAMPLE_RATE)
        self._spectrum_mask = (
            (self._freqs >= SPECTRUM_MIN_HZ)
            & (self._freqs <= min(SPECTRUM_MAX_HZ, SAMPLE_RATE / 2.0))
        )
        self.reset()
        self._initialized = True

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def reset(self, reason: str = "session") -> None:
        """清空全部流式状态，保留只读 FFT 表。"""
        self._audio = np.zeros(0, dtype=np.float32)
        self.frames_processed = 0
        self.samples_processed = 0
        self.reset_count = (
            self.reset_count + 1
            if getattr(self, "_initialized", False) else 0
        )
        self.last_reset_reason = str(reason)

        self._noise_calibration: List[float] = []
        self.noise_floor_dbfs = -160.0
        self._calibrated = False
        self._active_run_frames = 0
        self._prev_db: Deque[float] = deque(maxlen=4)
        self._prev_spectrum: Optional[np.ndarray] = None
        self._prev_active = False

        self._tone_segment: Optional[_ToneSegment] = None
        self._tone_hold: Optional[AcousticFrame] = None
        self._tone_pulses: Deque[_TonePulse] = deque()
        self._last_tone_emit_s = -1.0e9

        self._pending_impacts: List[_PendingImpact] = []
        self._knock_cluster: Optional[_KnockCluster] = None
        self._knock_cluster_seq = 0
        self._last_onset_frame: Optional[int] = None

        self._last_frame: Optional[AcousticFrame] = None
        self._last_candidates: List[AcousticCandidate] = []

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def push(self, block: Sequence[float]) -> List[AcousticCandidate]:
        """喂入一块音频；返回本次新确认的影子候选。"""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []
        self._audio = np.concatenate([self._audio, x.copy()])
        self.samples_processed += int(x.size)

        out: List[AcousticCandidate] = []
        while self._audio.size >= FRAME_SAMPLES:
            frame_samples = self._audio[:FRAME_SAMPLES]
            self._audio = self._audio[FRAME_SAMPLES:]
            out.extend(self._process_frame(frame_samples))
        self._last_candidates = out
        return out

    def snapshot(self) -> Dict[str, float | int | str | bool | None]:
        """返回当前特征状态；不返回音频。"""
        frame = self._last_frame
        return {
            "shadow_only": True,
            "frames_processed": int(self.frames_processed),
            "samples_processed": int(self.samples_processed),
            "noise_floor_dbfs": round(float(self.noise_floor_dbfs), 1),
            "calibrated": bool(self._calibrated),
            "current_relative_db": (
                round(float(frame.relative_db), 1) if frame is not None else None
            ),
            "current_spectral_flatness": (
                round(float(frame.spectral_flatness), 4) if frame is not None else None
            ),
            "current_spectral_centroid_hz": (
                round(float(frame.spectral_centroid_hz), 1)
                if frame is not None else None
            ),
            "current_spectral_similarity": (
                round(float(frame.spectral_similarity), 4)
                if frame is not None and frame.spectral_similarity is not None
                else None
            ),
            "current_tonal": bool(frame.tonal) if frame is not None else False,
            "pending_impacts": len(self._pending_impacts),
            "recent_impacts": (
                len(self._knock_cluster.impacts)
                if self._knock_cluster is not None else 0
            ),
            "last_reset_reason": self.last_reset_reason,
            "reset_count": int(self.reset_count),
        }

    # ------------------------------------------------------------------ #
    # 单帧处理
    # ------------------------------------------------------------------ #
    def _process_frame(self, samples: np.ndarray) -> List[AcousticCandidate]:
        out: List[AcousticCandidate] = []
        frame_index = self.frames_processed
        self.frames_processed += 1

        x = np.asarray(samples, dtype=np.float64)
        x = x - float(np.mean(x))
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        rms_db = self._dbfs(rms)
        peak_db = self._dbfs(peak)
        crest = peak / rms if rms > 1.0e-12 else 0.0

        spectrum, flatness, centroid, dominant = self._spectral_features(x)
        similarity: Optional[float] = None
        flux = 0.0
        if self._prev_spectrum is not None:
            similarity = self._cosine_similarity(spectrum, self._prev_spectrum)
            flux = self._spectral_flux(spectrum, self._prev_spectrum)

        energy_rise = (
            rms_db - float(np.median(self._prev_db))
            if self._prev_db else 0.0
        )

        if not self._calibrated:
            self._noise_calibration.append(rms_db)
            self.noise_floor_dbfs = float(
                np.percentile(self._noise_calibration, 20)
            )
            if len(self._noise_calibration) >= NOISE_CALIBRATION_FRAMES:
                self._calibrated = True
            self.noise_floor_dbfs = max(self.noise_floor_dbfs, -100.0)
        else:
            if rms_db > self.noise_floor_dbfs + NOISE_RISE_LIMIT_DB:
                self._active_run_frames += 1
            else:
                self._active_run_frames = 0
            self._update_noise_floor(rms_db)
        self.noise_floor_dbfs = max(self.noise_floor_dbfs, -100.0)

        relative_db = rms_db - self.noise_floor_dbfs
        active = self._calibrated and relative_db >= ACTIVITY_MARGIN_DB
        tonal = bool(
            active
            and flatness <= TONE_FLATNESS_MAX
            and TONE_MIN_CENTROID_HZ <= centroid <= TONE_MAX_CENTROID_HZ
            and (
                similarity is None
                or similarity >= TONE_SIMILARITY_MIN
            )
        )

        frame = AcousticFrame(
            frame_index=frame_index,
            start_time_s=round(frame_index * FRAME_SECONDS, 6),
            end_time_s=round((frame_index + 1) * FRAME_SECONDS, 6),
            rms_dbfs=round(rms_db, 2),
            noise_floor_dbfs=round(self.noise_floor_dbfs, 2),
            relative_db=round(relative_db, 2),
            crest_factor=round(crest, 3),
            spectral_flatness=round(flatness, 6),
            spectral_centroid_hz=round(centroid, 1),
            dominant_hz=round(dominant, 1),
            spectral_similarity=(
                round(similarity, 6) if similarity is not None else None
            ),
            spectral_flux=round(flux, 6),
            energy_rise_db=round(energy_rise, 2),
            active=active,
            tonal=tonal,
        )

        if self._calibrated:
            out.extend(self._update_tone(frame))
            out.extend(self._update_impacts(frame, spectrum))

        self._prev_db.append(rms_db)
        self._prev_spectrum = spectrum
        self._prev_active = active
        self._last_frame = frame
        return out

    # ------------------------------------------------------------------ #
    # 门铃：音调持续与重复音调
    # ------------------------------------------------------------------ #
    def _update_tone(self, frame: AcousticFrame) -> List[AcousticCandidate]:
        out: List[AcousticCandidate] = []
        seg = self._tone_segment

        if frame.tonal:
            if seg is None or frame.frame_index - seg.last_frame > TONE_MAX_GAP_FRAMES:
                if seg is not None:
                    self._finish_tone_segment(seg)
                hold = self._tone_hold
                if (
                    hold is None
                    or frame.frame_index - hold.frame_index > TONE_CONFIRM_FRAMES
                ):
                    self._tone_hold = frame
                    pulse_candidate = self._repeated_tone_candidate()
                    if pulse_candidate is not None:
                        out.append(pulse_candidate)
                    return out
                seg = _ToneSegment(
                    start_frame=hold.frame_index,
                    last_frame=frame.frame_index,
                    peak_relative_db=max(hold.relative_db, frame.relative_db),
                    centroid_sum=(
                        hold.spectral_centroid_hz + frame.spectral_centroid_hz
                    ),
                    similarity_sum=(
                        (hold.spectral_similarity or 1.0)
                        + (frame.spectral_similarity or 1.0)
                    ),
                    similarity_count=2,
                    flatness_sum=(
                        hold.spectral_flatness + frame.spectral_flatness
                    ),
                    frame_count=2,
                )
                self._tone_segment = seg
                self._tone_hold = None
            else:
                seg.last_frame = frame.frame_index
                seg.peak_relative_db = max(
                    seg.peak_relative_db, frame.relative_db
                )
                seg.centroid_sum += frame.spectral_centroid_hz
                seg.flatness_sum += frame.spectral_flatness
                if frame.spectral_similarity is not None:
                    seg.similarity_sum += frame.spectral_similarity
                    seg.similarity_count += 1
                seg.frame_count += 1

            duration = self._segment_duration(seg)
            confirm_s = seg.start_frame * FRAME_SECONDS + duration
            if (
                not seg.emitted
                and duration >= TONE_MIN_DURATION_S
                and confirm_s - self._last_tone_emit_s
                >= DOORBELL_EMIT_COOLDOWN_S
            ):
                candidate = self._doorbell_segment_candidate(seg)
                out.append(candidate)
                seg.emitted = True
                self._last_tone_emit_s = candidate.time_s
        elif seg is not None and frame.frame_index - seg.last_frame > TONE_MAX_GAP_FRAMES:
            self._finish_tone_segment(seg)
            self._tone_segment = None
            self._tone_hold = None
        elif not frame.tonal:
            self._tone_hold = None

        # 短音调不单独触发，但若在节奏窗口内重复，可作为重复音调候选。
        pulse_candidate = self._repeated_tone_candidate()
        if pulse_candidate is not None:
            out.append(pulse_candidate)
        return out

    def _finish_tone_segment(self, seg: _ToneSegment) -> None:
        duration = self._segment_duration(seg)
        if duration < TONE_MIN_DURATION_S:
            similarity = seg.similarity_sum / max(seg.similarity_count, 1)
            self._tone_pulses.append(
                _TonePulse(
                    start_time_s=seg.start_frame * FRAME_SECONDS,
                    duration_s=duration,
                    centroid_hz=seg.centroid_sum / max(seg.frame_count, 1),
                    similarity=similarity,
                    relative_db=seg.peak_relative_db,
                )
            )
            self._prune_tone_pulses()

    def _doorbell_segment_candidate(self, seg: _ToneSegment) -> AcousticCandidate:
        duration = self._segment_duration(seg)
        centroid = seg.centroid_sum / max(seg.frame_count, 1)
        flatness = seg.flatness_sum / max(seg.frame_count, 1)
        similarity = seg.similarity_sum / max(seg.similarity_count, 1)
        start_s = seg.start_frame * FRAME_SECONDS
        confirm_s = start_s + duration
        return AcousticCandidate(
            candidate_type="doorbell",
            start_time_s=round(start_s, 3),
            time_s=round(confirm_s, 3),
            duration_s=round(duration, 3),
            relative_db=round(seg.peak_relative_db, 1),
            reason=(
                f"检测到稳定音调，持续时间 {duration:.2f}s，"
                f"频谱相似度 {similarity:.3f}，高于噪声底 "
                f"{seg.peak_relative_db:.1f} dB"
            ),
            features={
                "trigger_kind": "stable_tonal_segment",
                "tone_frames": int(seg.frame_count),
                "spectral_flatness": round(flatness, 4),
                "spectral_centroid_hz": round(centroid, 1),
                "spectral_similarity": round(similarity, 4),
                "noise_floor_dbfs": round(
                    self._last_frame.noise_floor_dbfs
                    if self._last_frame is not None else self.noise_floor_dbfs,
                    1,
                ),
            },
        )

    def _repeated_tone_candidate(self) -> Optional[AcousticCandidate]:
        self._prune_tone_pulses()
        if len(self._tone_pulses) < REPEATED_TONE_MIN_COUNT:
            return None
        newest = self._tone_pulses[-1]
        if newest.start_time_s - self._last_tone_emit_s < KNOCK_EMIT_COOLDOWN_S:
            return None
        pulses = list(self._tone_pulses)
        pulses = [
            p for p in pulses
            if p.duration_s >= REPEATED_TONE_MIN_PULSE_DURATION_S
            and p.similarity >= TONE_SIMILARITY_MIN
        ]
        if len(pulses) < REPEATED_TONE_MIN_COUNT:
            return None
        first = pulses[0]
        intervals = [
            pulses[i].start_time_s - pulses[i - 1].start_time_s
            for i in range(1, len(pulses))
        ]
        valid_intervals = [
            x for x in intervals if KNOCK_MIN_INTERVAL_S <= x <= REPEATED_TONE_WINDOW_S
        ]
        if len(valid_intervals) < REPEATED_TONE_MIN_COUNT - 1:
            return None
        self._last_tone_emit_s = newest.start_time_s + newest.duration_s
        return AcousticCandidate(
            candidate_type="doorbell",
            start_time_s=round(first.start_time_s, 3),
            time_s=round(newest.start_time_s + newest.duration_s, 3),
            duration_s=round(
                newest.start_time_s + newest.duration_s - first.start_time_s, 3
            ),
            relative_db=round(max(p.relative_db for p in pulses), 1),
            reason=(
                f"检测到 {len(pulses)} 次重复音调，"
                f"节奏间隔 {valid_intervals[-1]:.2f}s"
            ),
            features={
                "trigger_kind": "repeated_tonal_pulses",
                "tone_pulse_count": len(pulses),
                "latest_pulse_duration_s": round(newest.duration_s, 3),
                "latest_interval_s": round(valid_intervals[-1], 3),
                "spectral_centroid_hz": round(newest.centroid_hz, 1),
            },
        )

    def _prune_tone_pulses(self) -> None:
        if not self._tone_pulses:
            return
        newest = self._tone_pulses[-1].start_time_s
        while (
            self._tone_pulses
            and newest - self._tone_pulses[0].start_time_s > REPEATED_TONE_WINDOW_S
        ):
            self._tone_pulses.popleft()

    # ------------------------------------------------------------------ #
    # 敲门：起音、衰减与连续冲击
    # ------------------------------------------------------------------ #
    def _update_impacts(
        self, frame: AcousticFrame, spectrum: np.ndarray
    ) -> List[AcousticCandidate]:
        out: List[AcousticCandidate] = []
        self._expire_knock_cluster(frame.start_time_s)

        # 先推进并确认旧冲击，避免当前帧同时参与自己的衰减判断。
        remaining: List[_PendingImpact] = []
        for impact in self._pending_impacts:
            age = (frame.frame_index - impact.start_frame) * FRAME_SECONDS
            impact.last_db = frame.rms_dbfs
            impact.frame_count += 1
            impact.centroid_sum += frame.spectral_centroid_hz
            impact.flatness_sum += frame.spectral_flatness
            if frame.spectral_similarity is not None:
                impact.similarity_sum += frame.spectral_similarity
                impact.similarity_count += 1
                impact.similarity_max = max(
                    impact.similarity_max, frame.spectral_similarity
                )

            decay = impact.peak_db - frame.rms_dbfs
            decayed = decay >= IMPACT_DECAY_DB
            too_long = age >= IMPACT_MAX_DURATION_S
            enough_analysis = impact.frame_count >= IMPACT_CONFIRM_FRAMES
            if (
                enough_analysis
                and age >= IMPACT_MIN_DURATION_S
                and (decayed or too_long)
            ):
                confirmed = self._finalize_impact(impact, confirmed_by_time=too_long)
                if confirmed:
                    out.extend(self._add_confirmed_impact(confirmed))
            else:
                remaining.append(impact)
        self._pending_impacts = remaining

        # 当前帧出现新的起音。
        onset = self._is_impact_onset(frame)
        if onset:
            self._pending_impacts.append(
                _PendingImpact(
                    start_frame=frame.frame_index,
                    onset_relative_db=frame.relative_db,
                    rise_db=frame.energy_rise_db,
                    spectral_flux=frame.spectral_flux,
                    peak_db=frame.rms_dbfs,
                    peak_frame=frame.frame_index,
                    last_db=frame.rms_dbfs,
                    centroid_sum=frame.spectral_centroid_hz,
                    similarity_sum=(
                        frame.spectral_similarity
                        if frame.spectral_similarity is not None else 1.0
                    ),
                    similarity_count=1,
                    similarity_max=(
                        frame.spectral_similarity
                        if frame.spectral_similarity is not None else 0.0
                    ),
                    flatness_sum=frame.spectral_flatness,
                    frame_count=1,
                )
            )

        self._expire_knock_cluster(frame.end_time_s)
        return out

    def _is_impact_onset(self, frame: AcousticFrame) -> bool:
        if not frame.active or frame.relative_db < ONSET_MARGIN_DB:
            return False
        if frame.energy_rise_db < IMPACT_MIN_RISE_DB:
            return False
        if frame.spectral_flux < ONSET_SPECTRAL_FLUX_MIN:
            return False
        if not (
            IMPACT_MIN_CENTROID_HZ
            <= frame.spectral_centroid_hz
            <= IMPACT_MAX_CENTROID_HZ
        ):
            return False
        if (
            self._last_onset_frame is not None
            and (frame.frame_index - self._last_onset_frame) * FRAME_SECONDS
            < ONSET_REFRACTORY_S
        ):
            return False
        # 同一个上升沿只记一次。
        is_new = not any(
            (frame.frame_index - p.start_frame) * FRAME_SECONDS
            < KNOCK_MIN_INTERVAL_S
            for p in self._pending_impacts
        )
        if is_new:
            self._last_onset_frame = frame.frame_index
        return is_new

    def _finalize_impact(
        self, impact: _PendingImpact, confirmed_by_time: bool
    ) -> Optional[_ConfirmedImpact]:
        duration = (impact.peak_frame - impact.start_frame + 1) * FRAME_SECONDS
        decay = impact.peak_db - impact.last_db
        if duration < IMPACT_MIN_DURATION_S:
            return None
        if duration > IMPACT_MAX_DURATION_S + FRAME_SECONDS:
            return None
        if not confirmed_by_time and decay < IMPACT_DECAY_DB:
            return None
        if duration >= IMPACT_MAX_DURATION_S and decay < IMPACT_DECAY_DB:
            return None

        centroid = impact.centroid_sum / max(impact.frame_count, 1)
        flatness = impact.flatness_sum / max(impact.frame_count, 1)
        similarity = (
            impact.similarity_sum / impact.similarity_count
            if impact.similarity_count else None
        )
        # 稳定门铃/报警音更像持续音，不当作敲门冲击。
        if (
            impact.similarity_max >= TONE_SIMILARITY_MIN
            and flatness <= TONE_FLATNESS_MAX
            and centroid >= TONE_MIN_CENTROID_HZ
        ):
            return None

        return _ConfirmedImpact(
            start_time_s=impact.start_frame * FRAME_SECONDS,
            time_s=(impact.peak_frame + 1) * FRAME_SECONDS,
            duration_s=duration,
            relative_db=impact.peak_db - self.noise_floor_dbfs,
            centroid_hz=centroid,
            similarity=similarity,
            flatness=flatness,
            decay_db=max(decay, 0.0),
        )

    def _add_confirmed_impact(
        self, confirmed: _ConfirmedImpact
    ) -> List[AcousticCandidate]:
        cluster = self._knock_cluster
        if (
            cluster is None
            or confirmed.start_time_s
            - max(x.start_time_s for x in cluster.impacts)
            > KNOCK_MAX_INTERVAL_S
        ):
            self._knock_cluster_seq += 1
            cluster = _KnockCluster(
                cluster_id=f"knock-{self._knock_cluster_seq}",
                impacts=[confirmed],
            )
            self._knock_cluster = cluster
        else:
            cluster.impacts.append(confirmed)
            cluster.impacts.sort(key=lambda x: x.start_time_s)

        impacts = cluster.impacts
        if len(impacts) < KNOCK_MIN_COUNT:
            return []

        intervals = [
            impacts[i].start_time_s - impacts[i - 1].start_time_s
            for i in range(1, len(impacts))
        ]
        if not intervals:
            return []

        first = impacts[0]
        last = impacts[-1]
        intervals_sorted = sorted(intervals)
        median_interval = intervals_sorted[len(intervals_sorted) // 2]
        avg_duration = sum(x.duration_s for x in impacts) / len(impacts)
        avg_centroid = sum(x.centroid_hz for x in impacts) / len(impacts)
        avg_decay = sum(x.decay_db for x in impacts) / len(impacts)
        peak_relative = max(x.relative_db for x in impacts)

        return [
            AcousticCandidate(
                candidate_type="knock",
                start_time_s=round(first.start_time_s, 3),
                time_s=round(last.time_s, 3),
                duration_s=round(last.time_s - first.start_time_s, 3),
                relative_db=round(peak_relative, 1),
                reason=(
                    f"检测到 {len(impacts)} 次短冲击，"
                    f"中位间隔 {median_interval:.2f}s，冲击后快速衰减"
                ),
                features={
                    "trigger_kind": "repeated_short_impacts",
                    "impact_count": len(impacts),
                    "impact_count_display": min(len(impacts), KNOCK_MAX_COUNT),
                    "cluster_id": cluster.cluster_id,
                    "median_interval_s": round(median_interval, 3),
                    "average_impact_duration_s": round(avg_duration, 3),
                    "average_impact_decay_db": round(avg_decay, 1),
                    "spectral_centroid_hz": round(avg_centroid, 1),
                    "noise_floor_dbfs": round(self.noise_floor_dbfs, 1),
                },
            )
        ]

    def _expire_knock_cluster(self, now_s: float) -> None:
        """超过最大冲击间隔没有新冲击时结束当前簇。"""
        cluster = self._knock_cluster
        if cluster is None or not cluster.impacts:
            return
        last_start = max(x.start_time_s for x in cluster.impacts)
        if now_s - last_start > KNOCK_MAX_INTERVAL_S:
            self._knock_cluster = None

    # ------------------------------------------------------------------ #
    # 数值工具
    # ------------------------------------------------------------------ #
    def _spectral_features(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        powered = np.abs(np.fft.rfft(x * self._window)) ** 2
        selected = powered[self._spectrum_mask] + 1.0e-20
        frequencies = self._freqs[self._spectrum_mask]
        total = float(np.sum(selected))
        flatness = float(
            np.exp(np.mean(np.log(selected))) / (np.mean(selected) + 1.0e-20)
        )
        centroid = float(np.sum(frequencies * selected) / max(total, 1.0e-20))
        dominant = float(frequencies[int(np.argmax(selected))])
        shape = selected / max(total, 1.0e-20)
        return shape, flatness, centroid, dominant

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1.0e-20:
            return 0.0
        return float(np.clip(np.dot(a, b) / denominator, 0.0, 1.0))

    @staticmethod
    def _spectral_flux(current: np.ndarray, previous: np.ndarray) -> float:
        rise = np.maximum(current - previous, 0.0)
        return float(np.sum(rise) / max(float(np.sum(previous)), 1.0e-20))

    @staticmethod
    def _dbfs(value: float) -> float:
        if value <= 0.0:
            return -160.0
        return max(-160.0, 20.0 * float(np.log10(value)))

    def _update_noise_floor(self, rms_db: float) -> None:
        if rms_db < self.noise_floor_dbfs:
            alpha = NOISE_FALL_ALPHA
        elif rms_db <= self.noise_floor_dbfs + NOISE_RISE_LIMIT_DB:
            alpha = NOISE_RISE_ALPHA
        elif self._active_run_frames >= NOISE_ACTIVE_ADAPT_AFTER_FRAMES:
            alpha = NOISE_ACTIVE_RISE_ALPHA
        else:
            return
        self.noise_floor_dbfs = (
            (1.0 - alpha) * self.noise_floor_dbfs + alpha * rms_db
        )

    @staticmethod
    def _segment_duration(seg: _ToneSegment) -> float:
        return (seg.last_frame - seg.start_frame + 1) * FRAME_SECONDS


__all__ = [
    "AcousticShadowDetector",
    "AcousticFrame",
    "AcousticCandidate",
    "SAMPLE_RATE",
    "FRAME_SAMPLES",
    "FRAME_SECONDS",
]
