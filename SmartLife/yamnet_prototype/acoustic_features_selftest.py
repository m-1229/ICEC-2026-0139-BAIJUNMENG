#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""声学特征影子检测自测。

只使用合成波和现有只读 WAV；不打开麦克风、不写文件、不加载 YAMNet。

运行：
    yamnet_prototype\.venv\Scripts\python.exe -B acoustic_features_selftest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from acoustic_features import (  # noqa: E402
    FRAME_SAMPLES,
    AcousticCandidate,
    AcousticShadowDetector,
)
from yamnet_batch_eval import read_wav, resample_fft  # noqa: E402


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[tuple[str, str]] = []
        self.infos: list[tuple[str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed.append((name, detail))

    def info(self, name: str, detail: str) -> None:
        print(f"  [INFO] {name}  {detail}")
        self.infos.append((name, detail))


R = Report()
SR = 48000


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(SR * seconds), dtype=np.float32)


def noise(seconds: float, seed: int, amplitude: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(round(SR * seconds)) * amplitude).astype(np.float32)


def tone(freq: float, seconds: float, amplitude: float = 0.15) -> np.ndarray:
    t = np.arange(round(SR * seconds), dtype=np.float64) / SR
    return (np.sin(2.0 * np.pi * freq * t) * amplitude).astype(np.float32)


def impact(amplitude: float = 0.35, seconds: float = 0.20) -> np.ndarray:
    n = round(SR * seconds)
    t = np.arange(n, dtype=np.float64) / SR
    # 低频瞬态 + 短衰减包络，贴近轻敲/木门冲击的简单近似。
    env = np.exp(-t / 0.045)
    body = (
        np.sin(2.0 * np.pi * 180.0 * t)
        + 0.45 * np.sin(2.0 * np.pi * 420.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 950.0 * t)
    )
    return (body * env * amplitude / 1.65).astype(np.float32)


def run_blocks(
    signal: np.ndarray,
    chunk: int = FRAME_SAMPLES,
    lead_seconds: float = 0.30,
) -> tuple[list[AcousticCandidate], AcousticShadowDetector]:
    det = AcousticShadowDetector()
    wave = np.concatenate([silence(lead_seconds), np.asarray(signal, dtype=np.float32)])
    candidates: list[AcousticCandidate] = []
    for i in range(0, wave.size, chunk):
        upsert_candidates(candidates, det.push(wave[i:i + chunk]))
    # 给已开始的冲击足够的尾部帧完成衰减判断。
    upsert_candidates(candidates, det.push(silence(0.45)))
    return candidates, det


def upsert_candidates(
    current: list[AcousticCandidate],
    updates: list[AcousticCandidate],
) -> None:
    """模拟监听状态接口：同一 knock 簇只保留最新版本。"""
    for candidate in updates:
        if candidate.candidate_type != "knock":
            current.append(candidate)
            continue
        cluster_id = candidate.features.get("cluster_id")
        for index, existing in enumerate(current):
            if (
                existing.candidate_type == "knock"
                and existing.features.get("cluster_id") == cluster_id
            ):
                current[index] = candidate
                break
        else:
            current.append(candidate)


def kinds(candidates: list[AcousticCandidate]) -> set[str]:
    return {c.candidate_type for c in candidates}


def candidate_summary(candidates: list[AcousticCandidate]) -> str:
    return str([
        (c.candidate_type, c.time_s, c.features.get("trigger_kind"))
        for c in candidates
    ])


def load_48k_mono(path: str) -> np.ndarray:
    """读取 WAV 并转换为只读测试所需的 48 kHz 单声道波形。"""
    data, info = read_wav(path)
    mono = (
        data.mean(axis=1).astype(np.float64)
        if int(info["channels"]) > 1 else data[:, 0].astype(np.float64)
    )
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    mono = np.clip(mono, -1.0, 1.0)
    if int(info["sample_rate"]) != SR:
        mono = resample_fft(mono, int(info["sample_rate"]), SR)
    return np.ascontiguousarray(mono, dtype=np.float32)


def test_silence_and_noise() -> None:
    print("\n[A] 静音与普通噪声")
    for seconds, name in ((8.0, "8 秒静音"), (12.0, "12 秒普通噪声")):
        wave = silence(seconds) if seconds == 8.0 else noise(seconds, seed=17)
        candidates, det = run_blocks(wave, lead_seconds=0.0)
        R.check(
            name,
            not candidates,
            f"候选={candidate_summary(candidates)} 校准={det.snapshot()['calibrated']}",
        )


def test_stable_tone() -> None:
    print("\n[B] 稳定音调门铃")
    candidates, det = run_blocks(tone(880.0, 1.20))
    door = [c for c in candidates if c.candidate_type == "doorbell"]
    R.check("稳定 880 Hz 音调产生门铃候选", bool(door), candidate_summary(candidates))
    R.check("稳定长音调不因持续存在而反复刷候选",
            len(door) == 1, f"门铃候选={len(door)}")
    R.check("音调候选包含持续时间和频谱稳定性",
            bool(door) and door[0].duration_s >= 0.14
            and "spectral_similarity" in door[0].features,
            candidate_summary(candidates))
    snap = det.snapshot()
    R.check("最终状态包含噪声底和当前相对电平",
            "noise_floor_dbfs" in snap and "current_relative_db" in snap,
            str(snap))


def test_repeated_tones() -> None:
    print("\n[C] 重复音调门铃")
    short = tone(720.0, 0.08, amplitude=0.14)
    signal = np.concatenate([
        short,
        silence(0.22),
        short,
        silence(0.22),
        short,
    ])
    candidates, _ = run_blocks(signal)
    repeated = [
        c for c in candidates
        if c.candidate_type == "doorbell"
        and c.features.get("trigger_kind") == "repeated_tonal_pulses"
    ]
    R.check("三个短音调在窗口内形成重复音调候选", bool(repeated),
            candidate_summary(candidates))


def test_repeated_impacts() -> None:
    print("\n[D] 短冲击与连续敲门")
    one = impact()
    single, _ = run_blocks(one)
    R.check("单次孤立冲击不产生敲门候选",
            not any(c.candidate_type == "knock" for c in single),
            candidate_summary(single))

    burst = np.concatenate([one, silence(0.20), one, silence(0.20), one])
    candidates, _ = run_blocks(burst)
    knock = [c for c in candidates if c.candidate_type == "knock"]
    R.check("三次短冲击产生敲门候选", bool(knock), candidate_summary(candidates))
    R.check("敲门候选报告冲击次数和间隔",
            bool(knock) and int(knock[0].features.get("impact_count", 0)) >= 2
            and float(knock[0].features.get("median_interval_s", 0.0)) > 0.0,
            candidate_summary(candidates))
    R.check("同一冲击簇只保留一个候选",
            len(knock) == 1,
            f"knock候选={len(knock)} {candidate_summary(knock)}")
    R.check("同一簇后续冲击更新计数",
            bool(knock) and int(knock[0].features.get("impact_count", 0)) == 3,
            str(knock[0].features if knock else {}))

    ten = [impact()]
    for _ in range(9):
        ten.extend([silence(0.20), impact()])
    long_candidates, _ = run_blocks(np.concatenate(ten))
    long_knocks = [c for c in long_candidates if c.candidate_type == "knock"]
    R.check("超过 8 次冲击不会使整簇失效",
            len(long_knocks) == 1
            and int(long_knocks[0].features.get("impact_count", 0)) == 10
            and int(long_knocks[0].features.get("impact_count_display", 0)) == 8,
            str(long_knocks[0].features if long_knocks else {}))

    two_clusters = np.concatenate([
        impact(), silence(0.20), impact(),
        silence(2.0),
        impact(), silence(0.20), impact(),
    ])
    separated, _ = run_blocks(two_clusters)
    separated_knocks = [c for c in separated if c.candidate_type == "knock"]
    R.check("间隔超过1.5秒形成两个独立候选",
            len(separated_knocks) == 2
            and separated_knocks[0].time_s < separated_knocks[1].start_time_s,
            candidate_summary(separated_knocks))


def test_wrong_rhythm() -> None:
    print("\n[E] 错误节奏与持续噪声")
    short_impact = impact(seconds=0.04)
    too_close = np.concatenate([short_impact, silence(0.01), short_impact])
    close_candidates, _ = run_blocks(too_close)
    R.check("间隔过短的连续冲击不形成敲门候选",
            not any(c.candidate_type == "knock" for c in close_candidates),
            candidate_summary(close_candidates))

    too_far = np.concatenate([impact(), silence(2.0), impact()])
    far_candidates, _ = run_blocks(too_far)
    R.check("间隔过长的冲击不形成同一敲门节奏",
            not any(c.candidate_type == "knock" for c in far_candidates),
            candidate_summary(far_candidates))

    sustained = noise(8.0, seed=31, amplitude=0.08)
    sustained_candidates, _ = run_blocks(sustained)
    R.check("持续宽带噪声不产生候选",
            not sustained_candidates, candidate_summary(sustained_candidates))


def test_adaptive_noise_floor() -> None:
    print("\n[E2] 自适应噪声底")
    det = AcousticShadowDetector()
    low_noise = noise(2.0, seed=51, amplitude=0.0001)
    high_noise = noise(8.0, seed=52, amplitude=0.012)
    low_candidates = det.push(low_noise)
    low_floor = det.snapshot()["noise_floor_dbfs"]
    high_candidates = det.push(high_noise)
    high_floor = det.snapshot()["noise_floor_dbfs"]
    R.check("环境底噪升高后噪声底跟随",
            float(high_floor) > float(low_floor) + 10.0,
            f"{low_floor} dBFS -> {high_floor} dBFS")
    R.check("底噪变化过程不产生候选",
            not low_candidates and not high_candidates,
            candidate_summary(low_candidates + high_candidates))


def test_block_boundaries() -> None:
    print("\n[F] 音频块边界稳定性")
    burst = np.concatenate([impact(), silence(0.20), impact(), silence(0.20), impact()])
    results = []
    for chunk in (137, 480, FRAME_SAMPLES, 1537, 4096):
        candidates, _ = run_blocks(burst, chunk=chunk)
        signature = [
            (c.candidate_type, c.start_time_s, c.time_s,
             c.features.get("trigger_kind"))
            for c in candidates
        ]
        results.append(signature)
        print(f"    chunk={chunk}: {signature}")
    R.check("不同音频块边界候选序列一致",
            all(x == results[0] for x in results[1:]),
            f"{len(results)} 种块长")


def test_reset() -> None:
    print("\n[G] 会话重置")
    det = AcousticShadowDetector()
    det.push(np.concatenate([silence(1.0), tone(900.0, 0.5)]))
    before = det.frames_processed
    det.reset(reason="resync")
    snap = det.snapshot()
    fresh_candidates = det.push(silence(0.25))
    R.check("reset 清空帧计数和候选缓冲",
            before > 0 and snap["frames_processed"] == 0
            and snap["recent_impacts"] == 0 and not fresh_candidates,
            str(snap))
    R.check("reset 记录重同步原因",
            snap["last_reset_reason"] == "resync" and snap["reset_count"] == 1,
            str(snap))


def test_existing_wav() -> None:
    print("\n[H] 现有 WAV 只读观察（不是准确率）")
    samples = (
        ("recordings/门铃/doorbell_classic(1).wav", "doorbell"),
        ("recordings/门铃/doorbell_large.wav", "doorbell"),
        ("recordings/门铃/doorbell_mechanical.wav", "doorbell"),
        ("recordings/门铃/门铃.wav", "doorbell"),
        ("recordings/敲门/knock_heavy.wav", "knock"),
        ("recordings/敲门/knock_light.wav", "knock"),
        ("recordings/敲门/knock_medium.wav", "knock"),
        ("recordings/敲门/敲门.wav", "knock"),
        ("recordings/烟雾报警/smoke_alarm_long.wav", "control"),
        ("recordings/烟雾报警/smoke_alarm_standard.wav", "control"),
        ("recordings/水壶/kettle_whistle.wav", "control"),
        ("recordings/婴儿哭声/baby_crying.wav", "control"),
    )
    for rel, expected in samples:
        mono = load_48k_mono(os.path.join(HERE, rel))
        candidates, _ = run_blocks(mono, lead_seconds=0.30)
        matching = [c for c in candidates if c.candidate_type == expected]
        R.info(
            os.path.basename(rel),
            f"期望分支={expected} 候选总数={len(candidates)} "
            f"匹配={len(matching)} {candidate_summary(candidates[:4])}",
        )


def test_target_materials() -> None:
    print("\n[I] 8 个目标素材回归（前置 0.3 秒低电平）")
    doorbells = (
        ("recordings/门铃/门铃.wav", 4),
        ("recordings/门铃/doorbell_classic(1).wav", 1),
        ("recordings/门铃/doorbell_large.wav", 4),
        ("recordings/门铃/doorbell_mechanical.wav", 2),
    )
    for rel, expected_count in doorbells:
        candidates, _ = run_blocks(load_48k_mono(os.path.join(HERE, rel)))
        bell = [c for c in candidates if c.candidate_type == "doorbell"]
        knock = [c for c in candidates if c.candidate_type == "knock"]
        R.check(
            f"门铃回归 {os.path.basename(rel)}",
            len(bell) == expected_count and not knock,
            f"doorbell={len(bell)} knock={len(knock)} "
            f"{candidate_summary(candidates)}",
        )

    knocks = (
        ("recordings/敲门/敲门.wav", 4),
        ("recordings/敲门/knock_light.wav", 1),
        ("recordings/敲门/knock_medium.wav", 1),
        ("recordings/敲门/knock_heavy.wav", 1),
    )
    for rel, expected_clusters in knocks:
        candidates, _ = run_blocks(load_48k_mono(os.path.join(HERE, rel)))
        found = [c for c in candidates if c.candidate_type == "knock"]
        ids = [c.features.get("cluster_id") for c in found]
        sorted_ok = all(
            found[i - 1].time_s < found[i].start_time_s
            for i in range(1, len(found))
        )
        R.check(
            f"敲门回归 {os.path.basename(rel)}",
            len(found) == expected_clusters
            and len(ids) == len(set(ids))
            and sorted_ok,
            f"候选={len(found)} ids={ids} "
            f"counts={[c.features.get('impact_count') for c in found]} "
            f"{candidate_summary(found)}",
        )

    heavy, _ = run_blocks(load_48k_mono(
        os.path.join(HERE, "recordings/敲门/knock_heavy.wav")
    ))
    heavy_knocks = [c for c in heavy if c.candidate_type == "knock"]
    R.check(
        "knock_heavy 单候选允许通过且计数合理",
        len(heavy_knocks) == 1
        and int(heavy_knocks[0].features.get("impact_count", 0)) >= 2,
        str(heavy_knocks[0].features if heavy_knocks else {}),
    )


def test_startup_control() -> None:
    print("\n[J] 启动校准对照（从 0 秒直接送入）")
    expected = (
        ("recordings/门铃/门铃.wav", 4),
        ("recordings/门铃/doorbell_classic(1).wav", 0),
        ("recordings/门铃/doorbell_large.wav", 3),
        ("recordings/门铃/doorbell_mechanical.wav", 0),
    )
    for rel, expected_count in expected:
        candidates, _ = run_blocks(
            load_48k_mono(os.path.join(HERE, rel)),
            lead_seconds=0.0,
        )
        bell = [c for c in candidates if c.candidate_type == "doorbell"]
        R.check(
            f"裸起对照 {os.path.basename(rel)}",
            len(bell) == expected_count,
            f"doorbell={len(bell)} {candidate_summary(bell)}",
        )

    for rel in (
        "recordings/敲门/敲门.wav",
        "recordings/敲门/knock_light.wav",
        "recordings/敲门/knock_medium.wav",
        "recordings/敲门/knock_heavy.wav",
    ):
        candidates, _ = run_blocks(
            load_48k_mono(os.path.join(HERE, rel)),
            lead_seconds=0.0,
        )
        knocks = [c for c in candidates if c.candidate_type == "knock"]
        R.check(
            f"裸起敲门仍命中 {os.path.basename(rel)}",
            bool(knocks),
            candidate_summary(knocks),
        )


def main() -> int:
    print("=" * 78)
    print("SmartLife 声学特征影子检测自测")
    print("=" * 78)
    test_silence_and_noise()
    test_stable_tone()
    test_repeated_tones()
    test_repeated_impacts()
    test_wrong_rhythm()
    test_adaptive_noise_floor()
    test_block_boundaries()
    test_reset()
    test_target_materials()
    test_startup_control()
    test_existing_wav()
    print("\n" + "=" * 78)
    print(f"PASS {R.passed}  FAIL {len(R.failed)}  INFO {len(R.infos)}")
    if R.failed:
        print("未通过项：")
        for name, detail in R.failed:
            print(f"  - {name}: {detail}")
    else:
        print("未通过项：无")
    print("结论:", "PASS" if not R.failed else "FAIL")
    return 0 if not R.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
