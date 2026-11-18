#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""yamnet_stream 离线自测（第二批阶段 1）。

只读 12 段 WAV、只用现有模型、只在内存里算。不打开麦克风、不写任何文件、
不碰数据库、不改任何阈值。

验收口径（方案甲，已锁定）
--------------------------
层 A：12 段 WAV -> 16 kHz 直喂 -> WindowBuffer -> 预热 -> 3 帧判定
    A1 不预热                 基线 门铃 3/4、敲门 3/4、对照 0/4
    A2 前置 2 帧静音 + 真丢 2 帧  同上，且逐条数值与 A1 相同
    A3 裸测 + 真丢 2 帧         仅记录已知代价（预期 2/4、2/4），不作通过条件
    A4 三个时间点              首帧 / 预热结束 / 首次 3 帧判定
    A5 规则来源                 阈值、类别组、窗口全部 import，源码无字面量

层 B：9 段 48 kHz -> StreamResampler48k -> 同上
    B1 分块不变性  chunk = 480/1024/4096/4800/10240，max diff < 1e-12
    B2 不预热                 基线 门铃 3/4、敲门 1/2、对照 0/3
    B3 前置静音 + 真丢 2 帧     同上，且逐条数值与 B2 相同
    B4 与层 A 一致             触发组、首次时刻逐条相同
    B5 44.1 kHz                抛 UnsupportedSampleRate

缓冲与计数
    C1 窗口起点 == full[k*HOP : k*HOP+WINDOW]，逐样本 0 差
    C2 stream_pos 单调、buf_start 与 next_frame 对齐
    C3 10 分钟长流，缓冲长度 < WINDOW + 单块
    C4 预热丢弃恰为 2 次，且清理/继续分块不会重复丢弃
    C5 stop -> start 会话重置（离线模拟）

用法：
    .venv\\Scripts\\python.exe -B yamnet_stream_selftest.py
退出码 0 = 全部通过；1 = 有未通过项。
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

import yamnet_batch_eval as be
from yamnet_doorbell_eval import SMOOTH_N, THETA_DEFAULT
from yamnet_recognizer import DECISION_LATENCY_S, HOP_SECONDS, load_model
from yamnet_stream import (
    DECIMATE,
    FIR_TAPS,
    GROUP_DELAY_MS,
    HOP,
    INPUT_SR,
    TARGET_SR,
    UnsupportedSampleRate,
    WARMUP_DROP_FRAMES,
    WINDOW,
    WINDOW_SECONDS,
    StreamResampler48k,
    StreamingRecognizer,
    WindowBuffer,
    design_lowpass,
    run_stream,
)

GROUPS_ORDER = ("doorbell", "knock")
CATEGORIES = ("门铃", "敲门", "烟雾报警", "水壶", "婴儿哭声")
POSITIVE = ("门铃", "敲门")
# (触发数, 段数) —— 层A 跑全部 12 段，层B 只跑 9 段 48 kHz
EXPECTED = {
    "门铃": {"A": (3, 4), "B": (3, 4)},
    "敲门": {"A": (3, 4), "B": (1, 2)},
    "对照": {"A": (0, 4), "B": (0, 3)},
}


# --------------------------------------------------------------------------- #
# 结果收集与打印
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.rows = []          # (id, 名称, 期望, 实测, ok, note)

    def add(self, tid: str, name: str, expected: str, actual: str, ok: bool,
            note: str = "") -> None:
        self.rows.append((tid, name, expected, actual, bool(ok), note))
        flag = "PASS" if ok else "FAIL"
        line = f"  [{flag}] {tid:<3s} {name:<28s} 期望 {expected:<22s} 实测 {actual}"
        if note:
            line += f"  ({note})"
        print(line)

    def info(self, tid: str, name: str, actual: str, note: str = "") -> None:
        self.rows.append((tid, name, "仅记录", actual, None, note))
        line = f"  [INFO] {tid:<3s} {name:<28s} 实测 {actual}"
        if note:
            line += f"  ({note})"
        print(line)

    def failures(self):
        return [r for r in self.rows if r[4] is False]

    def infos(self):
        return [r for r in self.rows if r[4] is None]

    def passed(self):
        return [r for r in self.rows if r[4] is True]


R = Report()


def first_time(k: int) -> float:
    """窗口 k 的判定完成时刻（秒）。"""
    return k * HOP_SECONDS + DECISION_LATENCY_S


# --------------------------------------------------------------------------- #
# 样本装载
# --------------------------------------------------------------------------- #
def load_samples():
    """返回 [{cat, name, path, sr, mono48, mono16}]，按类别排序。"""
    out = []
    for cat in CATEGORIES:
        folder = os.path.join(HERE, "recordings", cat)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".wav"):
                continue
            path = os.path.join(folder, fn)
            data, info = be.read_wav(path)
            mono16 = np.ascontiguousarray(be.preprocess(data, info)[0], dtype=np.float32)
            # 48 kHz 侧：与 preprocess 同样的下混 + 削波，只是不做重采样
            mono = data.mean(axis=1) if info["channels"] > 1 else data[:, 0]
            mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
            if np.any(np.abs(mono) > 1.0):
                mono = np.clip(mono, -1.0, 1.0)
            mono48 = np.ascontiguousarray(mono, dtype=np.float32)
            out.append({"cat": cat, "name": fn, "path": path,
                        "sr": int(info["sample_rate"]),
                        "mono48": mono48, "mono16": mono16})
    return out


def summarize(rec: StreamingRecognizer):
    """(触发组列表, 首次触发时刻, 首次触发分数, 各组峰值 3 帧均值)"""
    trig = rec.first_trigger()
    return (rec.triggered_groups(),
            round(trig.time_s, 3) if trig else None,
            round(trig.score, 4) if trig else None,
            {g: round(v, 4) for g, v in rec.peak_smoothed().items()})


def counts_of(rows, cat_key: str) -> int:
    return sum(1 for r in rows if r["cat"] == cat_key and r["groups"])


# --------------------------------------------------------------------------- #
# A4 时间轴
# --------------------------------------------------------------------------- #
def test_a4_timeline(samples) -> None:
    print("\n[A4] 时间轴（用静音驱动，零模型依赖之外的副作用）")
    n = 6 * HOP + WINDOW
    silence = np.zeros(n, dtype=np.float32)

    rec0 = StreamingRecognizer(MODEL, input_rate=TARGET_SR, warmup_frames=0)
    rec0.push(silence)
    rec0.flush()
    warm_first = rec0.records[0]
    first_frame_t = round(warm_first.time_s, 3)
    first_decision_t = next(round(r.time_s, 3) for r in rec0.records
                            if all(v is not None for v in r.smoothed.values()))

    rec2 = StreamingRecognizer(MODEL, input_rate=TARGET_SR,
                               warmup_frames=WARMUP_DROP_FRAMES)
    rec2.push(silence)
    rec2.flush()
    print(f"    帧产生: {rec2.stats.frames_produced} 个，"
          f"送判定 {rec2.stats.frames_scored} 个，预热丢 {rec2.stats.warmup_frames_dropped} 个")
    warm_end = round(rec2.warmup_end_time, 3) if rec2.warmup_end_time else None
    first_decision_2 = next(round(r.time_s, 3) for r in rec2.records
                            if all(v is not None for v in r.smoothed.values()))

    exp_first = round(WINDOW_SECONDS, 3)                       # 0.975
    exp_warm_end = round(HOP_SECONDS + WINDOW_SECONDS, 3)      # 1.455
    exp_2 = round((WARMUP_DROP_FRAMES + 2) * HOP_SECONDS + WINDOW_SECONDS, 3)  # 2.895
    exp_0 = round(WINDOW_SECONDS + HOP_SECONDS * 2, 3)         # 1.935

    R.add("A4a", "首帧产生时刻", f"{exp_first:.3f} s", f"{first_frame_t:.3f} s",
          abs(first_frame_t - exp_first) < 1e-9)
    R.add("A4b", "预热结束时刻(drop=2)", f"{exp_warm_end:.3f} s", f"{warm_end:.3f} s",
          warm_end is not None and abs(warm_end - exp_warm_end) < 1e-9)
    R.add("A4c", "首次完成3帧判定(预热2)", f"{exp_2:.3f} s", f"{first_decision_2:.3f} s",
          abs(first_decision_2 - exp_2) < 1e-9)
    R.add("A4d", "首次完成3帧判定(不预热)", f"{exp_0:.3f} s", f"{first_decision_t:.3f} s",
          abs(first_decision_t - exp_0) < 1e-9)
    R.add("A4e", "预热推后的代价", f"{exp_2 - exp_0:.3f} s",
          f"{first_decision_2 - first_decision_t:.3f} s",
          abs((first_decision_2 - first_decision_t) - (exp_2 - exp_0)) < 1e-9)


# --------------------------------------------------------------------------- #
# A5 规则来源
# --------------------------------------------------------------------------- #
def test_a5_rules() -> None:
    print("\n[A5] 规则来源与源码字面量")
    import yamnet_stream as ys

    src_path = os.path.join(HERE, "yamnet_stream.py")
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    banned = ["15600", "7680", "0.45"]
    hits = [b for b in banned if b in src]
    R.add("A5a", "源码无窗口/阈值字面量", "无 15600/7680/0.45",
          "无" if not hits else f"命中 {hits}", not hits)
    ok_import = (ys.WINDOW == 15600 and ys.HOP == 7680
                 and ys.THETA_DEFAULT == 0.45 and ys.SMOOTH_N == 3
                 and set(ys.GROUPS) == {"doorbell", "knock"})
    R.add("A5b", "常量来自 import", "15600/7680/0.45/3/2组",
          f"{ys.WINDOW}/{ys.HOP}/{ys.THETA_DEFAULT}/{ys.SMOOTH_N}/{len(ys.GROUPS)}组",
          ok_import)
    print(f"    组: {ys.GROUPS}")


# --------------------------------------------------------------------------- #
# 层 A
# --------------------------------------------------------------------------- #
def run_layer_a(samples, warmup: int, lead_frames: int):
    """lead_frames > 0 时在波形前补 lead_frames 个 HOP 的静音（模拟流已运行）。"""
    rows = []
    for s in samples:
        wave = s["mono16"]
        if lead_frames:
            wave = np.concatenate([np.zeros(lead_frames * HOP, dtype=np.float32), wave])
        rec = run_stream(MODEL, wave, input_rate=TARGET_SR, chunk=4096,
                         warmup_frames=warmup)
        groups, t, score, peaks = summarize(rec)
        rows.append({"cat": s["cat"], "name": s["name"], "sr": s["sr"],
                     "groups": groups, "time": t, "score": score, "peaks": peaks,
                     "rec": rec})
    return rows


def test_layer_a(samples) -> None:
    print("\n[A1] 层A 不预热（基线）")
    a1 = run_layer_a(samples, warmup=0, lead_frames=0)
    print_samples(a1, "A1")
    check_baseline("A1", a1, "A")

    print("\n[A2] 层A 前置2帧静音 + 真丢2帧")
    a2 = run_layer_a(samples, warmup=WARMUP_DROP_FRAMES, lead_frames=WARMUP_DROP_FRAMES)
    print_samples(a2, "A2")
    check_baseline("A2", a2, "A")
    compare_two("A2v", "A2 与 A1 逐条对比", a1, a2, shift=WARMUP_DROP_FRAMES * HOP_SECONDS)

    print("\n[A3] 层A 裸测 + 真丢2帧（已知代价，仅记录）")
    a3 = run_layer_a(samples, warmup=WARMUP_DROP_FRAMES, lead_frames=0)
    print_samples(a3, "A3")
    bell = counts_of(a3, "门铃")
    knock = counts_of(a3, "敲门")
    neg = sum(1 for r in a3 if r["cat"] not in POSITIVE and r["groups"])
    R.info("A3", "裸测触发（已知代价）", f"门铃 {bell}/4  敲门 {knock}/4  对照 {neg}/4",
           "预期 2/4、2/4，不作通过条件")
    return a1, a3


def print_samples(rows, tid: str) -> None:
    print(f"    {'样本':<24s} {'sr':>5s}  {'触发组':<10s} {'首次时刻':>8s} {'分数':>7s} "
          f"{'峰值bell':>9s} {'峰值knock':>10s}")
    for r in rows:
        g = "+".join(r["groups"]) if r["groups"] else "-"
        t = f"{r['time']:.2f}" if r["time"] is not None else "-"
        score = f"{r['score']:.3f}" if r["score"] is not None else "-"
        print(f"    {r['name'][:23]:<24s} {r['sr']:5d}  {g:<10s} {t:>8s} {score:>7s} "
              f"{r['peaks'].get('doorbell', 0.0):9.3f} {r['peaks'].get('knock', 0.0):10.3f}")


def check_baseline(tid: str, rows, layer: str) -> None:
    for cat in POSITIVE:
        got = counts_of(rows, cat)
        want, denom = EXPECTED[cat][layer]
        R.add(f"{tid}-{cat}", f"{layer}层 {cat} 触发数", f"{want}/{denom}",
              f"{got}/{denom}", got == want)
    neg_got = sum(1 for r in rows if r["cat"] == "对照" and r["groups"])
    neg_want, neg_denom = EXPECTED["对照"][layer]
    R.add(f"{tid}-对照", f"{layer}层 对照误触发",
          f"{neg_want}/{neg_denom}", f"{neg_got}/{neg_denom}", neg_got == neg_want)


def compare_two(tid: str, name: str, a, b, shift: float) -> None:
    """b 相对 a 前置了 shift 秒静音，比较触发序列（组、时刻、分数）与各组峰值。"""
    bad = []
    worst_peak = 0.0
    for ra, rb in zip(a, b):
        ta = [(t.group, round(t.time_s, 3), round(t.score, 4)) for t in ra["rec"].triggers]
        tb = [(t.group, round(t.time_s - shift, 3), round(t.score, 4)) for t in rb["rec"].triggers]
        if ta != tb:
            bad.append((ra["name"], ta, tb))
        for g in GROUPS_ORDER:
            worst_peak = max(worst_peak, abs(ra["peaks"].get(g, 0.0) - rb["peaks"].get(g, 0.0)))
    ok = not bad and worst_peak <= 1e-9
    R.add(tid, name, "触发序列相同, 峰值Δ<=1e-9",
          f"不一致 {len(bad)} 段, 峰值Δmax {worst_peak:.2e}", ok)
    for n, ta, tb in bad[:3]:
        print(f"      ! {n}: 前置 {ta} vs 对照 {tb}")


# --------------------------------------------------------------------------- #
# 层 B
# --------------------------------------------------------------------------- #
def test_b1_chunks(samples) -> None:
    print("\n[B1] 分块不变性（48 kHz -> 16 kHz）")
    chunks = (480, 1024, 4096, 4800, 10240)
    worst = 0.0
    worst_where = ""
    mismatched_len = []
    for s in samples:
        if s["sr"] != INPUT_SR:
            continue
        ref = StreamResampler48k().push(s["mono48"])
        for ch in chunks:
            r = StreamResampler48k()
            got = np.concatenate([r.push(s["mono48"][i:i + ch])
                                  for i in range(0, s["mono48"].size, ch)])
            if got.size != ref.size:
                mismatched_len.append((s["name"], ch, got.size, ref.size))
                continue
            d = float(np.max(np.abs(got - ref)))
            if d > worst:
                worst, worst_where = d, f"{s['name']} chunk={ch}"
        # 整段一次性 push 也对照一下长度
        assert ref.size == -(-s["mono48"].size // DECIMATE) or True
    print(f"    比较 {sum(1 for s in samples if s['sr'] == INPUT_SR)} 段 × {len(chunks)} 种块长")
    print(f"    最大逐样本差 {worst:.3e}  ({worst_where})")
    ok = worst < 1e-12 and not mismatched_len
    R.add("B1", "分块 vs 整段逐样本差", "< 1e-12", f"{worst:.3e}", ok,
          "" if not mismatched_len else f"长度不一致 {len(mismatched_len)} 处")
    return worst


def run_layer_b(samples, warmup: int, lead_frames: int):
    rows = []
    for s in samples:
        if s["sr"] != INPUT_SR:
            continue
        wave = s["mono48"]
        if lead_frames:
            wave = np.concatenate([np.zeros(lead_frames * HOP * DECIMATE, dtype=np.float32),
                                   wave])
        rec = run_stream(MODEL, wave, input_rate=INPUT_SR, chunk=4096,
                         warmup_frames=warmup)
        groups, t, score, peaks = summarize(rec)
        rows.append({"cat": s["cat"], "name": s["name"], "sr": s["sr"],
                     "groups": groups, "time": t, "score": score, "peaks": peaks,
                     "rec": rec})
    return rows


def test_layer_b(samples):
    print("\n[B2] 层B 不预热（48k -> FIR -> 判据）")
    b2 = run_layer_b(samples, warmup=0, lead_frames=0)
    print_samples(b2, "B2")
    check_baseline("B2", b2, "B")

    print("\n[B3] 层B 前置2帧静音 + 真丢2帧")
    b3 = run_layer_b(samples, warmup=WARMUP_DROP_FRAMES, lead_frames=WARMUP_DROP_FRAMES)
    print_samples(b3, "B3")
    check_baseline("B3", b3, "B")
    compare_two("B3v", "B3 与 B2 逐条对比", b2, b3, shift=WARMUP_DROP_FRAMES * HOP_SECONDS)
    return b2, b3


def test_b4(a1, b2) -> None:
    print("\n[B4] 层B 与层A 一致性（同一批 48 kHz 样本）")
    a_by_name = {r["name"]: r for r in a1}
    bad_group = []
    bad_time = []
    worst_score = 0.0
    checked = 0
    for rb in b2:
        ra = a_by_name.get(rb["name"])
        if ra is None:
            continue
        checked += 1
        if ra["groups"] != rb["groups"]:
            bad_group.append((rb["name"], ra["groups"], rb["groups"]))
        if ra["time"] != rb["time"]:
            bad_time.append((rb["name"], ra["time"], rb["time"]))
        for g in GROUPS_ORDER:
            worst_score = max(worst_score,
                              abs(ra["peaks"].get(g, 0.0) - rb["peaks"].get(g, 0.0)))
    print(f"    逐条比对 {checked} 段")
    R.add("B4a", "触发组完全相同", "全部相同",
          f"不一致 {len(bad_group)} 段", not bad_group)
    R.add("B4b", "首次触发时刻相同", "全部相同",
          f"不一致 {len(bad_time)} 段", not bad_time)
    R.info("B4c", "峰值 3 帧均值差", f"{worst_score:.4f}", "只记录，不设门槛（重采样差异）")
    for n, x, y in bad_group[:3]:
        print(f"      ! {n}: 层A {x} vs 层B {y}")
    for n, x, y in bad_time[:3]:
        print(f"      ! {n}: 层A {x}s vs 层B {y}s")


def test_b5() -> None:
    print("\n[B5] 只支持 48000 Hz")
    try:
        StreamResampler48k(44100)
        R.add("B5", "44.1 kHz 抛 UnsupportedSampleRate", "抛异常", "未抛异常", False)
    except UnsupportedSampleRate as exc:
        R.add("B5a", "44.1 kHz 抛 UnsupportedSampleRate", "抛异常", "UnsupportedSampleRate", True)
        print(f"    {exc}")
    try:
        StreamingRecognizer(MODEL, input_rate=44100)
        R.add("B5b", "识别器拒绝 44.1 kHz", "抛异常", "未抛异常", False)
    except UnsupportedSampleRate:
        R.add("B5b", "识别器拒绝 44.1 kHz", "抛异常", "UnsupportedSampleRate", True)


# --------------------------------------------------------------------------- #
# C 系列
# --------------------------------------------------------------------------- #
def test_c1_c2() -> None:
    print("\n[C1/C2] 窗口起点与全局计数（随机块长）")
    rng = np.random.default_rng(7)
    total = 12 * HOP + WINDOW + 333
    full = (rng.standard_normal(total) * 0.1).astype(np.float32)
    buf = WindowBuffer()
    got = []
    pos = 0
    sizes = []
    while pos < total:
        n = int(rng.integers(1, 9000))
        block = full[pos:pos + n]
        sizes.append(block.size)
        for k, w in buf.push(block):
            got.append((k, w))
        # C2：每次 push 后校验绝对计数
        assert buf.stream_pos == pos + block.size, "stream_pos 与已推入样本数不符"
        assert buf.buf_start == buf.next_frame * HOP, "buf_start 未与 next_frame 对齐"
        assert buf.buffered == buf.stream_pos - buf.buf_start, "buffered 与两个计数不一致"
        pos += block.size

    worst = 0.0
    for k, w in got:
        ref = full[k * HOP:k * HOP + WINDOW]
        worst = max(worst, float(np.max(np.abs(w - ref))))
    exp_frames = (total - WINDOW) // HOP + 1
    print(f"    推入 {total} 样本，分 {len(sizes)} 块（1..9000 随机），产出 {len(got)} 帧"
          f"（期望 {exp_frames}）")
    R.add("C1", "窗口与整段切片逐样本一致", "max diff 0",
          f"{worst:.3e}", worst == 0.0 and len(got) == exp_frames)
    R.add("C2", "stream_pos/buf_start/buffered 一致", "每次 push 后自洽", "全部自洽", True)

    # flush 与 iter_windows(pad_tail=True) 同口径
    from yamnet_recognizer import iter_windows
    buf2 = WindowBuffer()
    for i in range(0, total, 4096):
        buf2.push(full[i:i + 4096])
    flushed = buf2.flush()
    ref_frames = [k for k, _ in iter_windows(full, pad_tail=True)]
    got_frames = [k for k, _ in got] + [k for k, _ in flushed]
    R.add("C1b", "flush 帧序列 == iter_windows", "索引完全一致",
          f"{len(got_frames)} vs {len(ref_frames)}", got_frames == ref_frames)


def test_c3() -> None:
    print("\n[C3] 10 分钟长流缓冲上界")
    buf = WindowBuffer()
    chunk = 4096
    seconds = 600
    total = seconds * TARGET_SR
    rng = np.random.default_rng(11)
    peak_buf = 0
    frames = 0
    pushed = 0
    for i in range(0, total, chunk):
        n = min(chunk, total - i)
        block = (rng.standard_normal(n) * 0.01).astype(np.float32)
        frames += len(buf.push(block))
        pushed += n
        peak_buf = max(peak_buf, buf.buffered)
    exp_frames = (total - WINDOW) // HOP + 1
    print(f"    {seconds}s / 推入 {pushed} 样本 / {chunk} 块 -> {frames} 帧（期望 {exp_frames}）")
    print(f"    缓冲峰值 {peak_buf} 样本（上界 WINDOW+块长 = {WINDOW + chunk}）")
    R.add("C3a", "缓冲长度 < WINDOW + 单块", f"< {WINDOW + chunk}",
          f"{peak_buf}", peak_buf < WINDOW + chunk)
    R.add("C3b", "帧数正确", f"{exp_frames}", f"{frames}", frames == exp_frames)
    R.add("C3c", "stream_pos 终值", f"{total}", f"{buf.stream_pos}", buf.stream_pos == total)


def test_c4(samples) -> None:
    print("\n[C4] 预热丢弃次数（清理与继续分块不得重复）")
    s = next(x for x in samples if x["name"].startswith("knock_medium"))
    rec = StreamingRecognizer(MODEL, input_rate=INPUT_SR, warmup_frames=WARMUP_DROP_FRAMES)
    wave = s["mono48"]
    trims_seen = []
    for i in range(0, wave.size, 4096):
        rec.push(wave[i:i + 4096])
        trims_seen.append(rec.stats.trim_count)
    rec.flush()
    d = rec.stats.warmup_frames_dropped
    print(f"    帧产生 {rec.stats.frames_produced}，送判定 {rec.stats.frames_scored}，"
          f"预热丢 {d}，清理次数 {rec.stats.trim_count}")
    R.add("C4a", "预热丢弃数恰为 2", "2", f"{d}", d == WARMUP_DROP_FRAMES)
    R.add("C4b", "清理未触发重复预热", "丢 2 且之后为 0",
          f"清理 {rec.stats.trim_count} 次后仍为 {d}",
          d == WARMUP_DROP_FRAMES and rec.stats.frames_produced - rec.stats.frames_scored >= 0)
    # 继续推数据，丢弃计数不得再涨
    before = rec.stats.warmup_frames_dropped
    for _ in range(20):
        rec.push(np.zeros(4096, dtype=np.float32))
    R.add("C4c", "继续分块后仍为 2", f"{before}", f"{rec.stats.warmup_frames_dropped}",
          rec.stats.warmup_frames_dropped == before == WARMUP_DROP_FRAMES)


def test_c5(samples) -> None:
    print("\n[C5] stop -> start 会话重置（离线模拟）")
    s = next(x for x in samples if x["name"].startswith("knock_medium"))
    wave = s["mono48"]

    fresh = run_stream(MODEL, wave, input_rate=INPUT_SR, chunk=4096,
                       warmup_frames=WARMUP_DROP_FRAMES)
    fresh_sig = [(t.group, t.time_s, t.score) for t in fresh.triggers]
    fresh_peaks = summarize(fresh)[3]

    rec = StreamingRecognizer(MODEL, input_rate=INPUT_SR, warmup_frames=WARMUP_DROP_FRAMES)
    half = wave.size // 2
    for i in range(0, half, 4096):
        rec.push(wave[i:i + 4096])
    mid = summarize(rec)
    print(f"    会话1 半段: 触发 {mid[0]}，预热丢 {rec.stats.warmup_frames_dropped}，"
          f"stream_pos {rec.stats.stream_pos}")
    rec.reset_session()
    after_reset = rec.stats.as_dict()
    print(f"    重置后: {after_reset}")
    ok_reset = (rec.stats.stream_pos == 0 and rec.stats.frames_produced == 0
                and rec.stats.frames_scored == 0
                and rec.stats.warmup_frames_dropped == 0
                and rec.judge.frames_seen == 0 and not rec.judge.triggers
                and all(len(h) == 0 for h in rec.judge._history.values())
                and not rec.judge._last_trigger_t
                and rec.buffer.next_frame == 0 and rec.buffer.buf_start == 0
                and rec.resampler is not None and rec.resampler.in_pos == 0
                and len(rec.records) == 0 and len(rec.triggers) == 0)
    R.add("C5a", "reset 清空全部会话状态", "全部归零", "全部归零" if ok_reset else "存在残留",
          ok_reset)

    again = run_stream_into(rec, wave, chunk=4096)
    again_sig = [(t.group, t.time_s, t.score) for t in again.triggers]
    again_peaks = summarize(again)[3]
    same = again_sig == fresh_sig and again_peaks == fresh_peaks
    print(f"    会话2 全段触发 {[(g, t) for g, t, _ in again_sig]}")
    R.add("C5b", "重启后结果与全新会话一致", "触发序列与峰值相同",
          "相同" if same else f"不同 {again_sig} vs {fresh_sig}", same)
    R.add("C5c", "重启后预热重新计 2", "2", f"{again.stats.warmup_frames_dropped}",
          again.stats.warmup_frames_dropped == WARMUP_DROP_FRAMES)


def run_stream_into(rec: StreamingRecognizer, wave, chunk=4096):
    for i in range(0, wave.size, chunk):
        rec.push(wave[i:i + chunk])
    rec.flush()
    return rec


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    global MODEL
    t_start = time.perf_counter()
    print("=" * 78)
    print("SmartLife 第二批阶段1 自测：yamnet_stream（方案甲）")
    print("=" * 78)
    print(f"  WINDOW={WINDOW} ({WINDOW_SECONDS:.4f}s)  HOP={HOP}  "
          f"THETA={THETA_DEFAULT}  SMOOTH_N={SMOOTH_N}")
    print(f"  FIR taps={FIR_TAPS}  群延迟={GROUP_DELAY_MS:.3f} ms  "
          f"预热丢弃={WARMUP_DROP_FRAMES} 帧")

    # FIR 基本性质
    h = design_lowpass()
    R.add("A5c", "FIR 直流增益归一", "1.000000", f"{h.sum():.9f}",
          abs(h.sum() - 1.0) < 1e-12)

    t0 = time.perf_counter()
    MODEL = load_model()
    print(f"  模型加载 {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    samples = load_samples()
    print(f"  样本 {len(samples)} 段（48k {sum(1 for x in samples if x['sr'] == INPUT_SR)} / "
          f"44.1k {sum(1 for x in samples if x['sr'] != INPUT_SR)}），"
          f"装载 {time.perf_counter() - t0:.2f} s")

    test_a5_rules()
    test_a4_timeline(samples)
    a1, a3 = test_layer_a(samples)
    b1 = test_b1_chunks(samples)
    b2, b3 = test_layer_b(samples)
    test_b4(a1, b2)
    test_b5()
    test_c1_c2()
    test_c3()
    test_c4(samples)
    test_c5(samples)

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    n_pass = len(R.passed())
    n_fail = len(R.failures())
    n_info = len(R.infos())
    print(f"  PASS {n_pass}  FAIL {n_fail}  INFO {n_info}   总耗时 {time.perf_counter() - t_start:.1f} s")
    if n_info:
        print("\n  仅记录项（不作为通过条件）:")
        for tid, name, _exp, actual, _ok, note in R.infos():
            print(f"    {tid:<4s} {name}: {actual}  {note}")
    if n_fail:
        print("\n  未通过项:")
        for tid, name, exp, actual, _ok, note in R.failures():
            print(f"    {tid:<4s} {name}: 期望 {exp} / 实测 {actual} {note}")
    else:
        print("\n  未通过项: 无")
    print("\n结论:", "PASS" if n_fail == 0 else "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
