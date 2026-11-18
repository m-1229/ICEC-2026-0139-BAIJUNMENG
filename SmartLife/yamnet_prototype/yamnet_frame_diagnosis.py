"""YAMNet frame-level diagnosis (stage 3).

Independent diagnostic tool. It reuses the *same* preprocessing as
yamnet_batch_eval.py (mono channel mean -> FFT resample to 16 kHz -> float32)
so the per-frame numbers stay comparable with the run-2 baseline report, but it
does not modify, import or depend on that script.

What it does, per WAV file:
  * resolves the target class names against the real assets/yamnet_class_map.csv
    (no class name is hard-coded on faith - every lookup is recorded)
  * runs frame-level inference (0.96 s window / 0.48 s hop)
  * reports every frame: time position, energy, top-5 classes, and the score of
    every target-related class
  * reports best target score + its time, frame counts above 0.3 / 0.5 / 0.7,
    and the longest *consecutive* run above each threshold
  * checks source clipping (read-only)

Wording: all numbers are model scores (independent per-class sigmoid outputs).
They are NOT accuracy and NOT calibrated confidence. A single frame above a
threshold is likewise NOT treated as a successful recognition - the verdict
requires a consecutive run.

Usage:
    python yamnet_frame_diagnosis.py
    python yamnet_frame_diagnosis.py --dir recordings --min-run 3
"""

import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(HERE, "models", "yamnet")
DEFAULT_AUDIO_DIR = os.path.join(HERE, "recordings")
DEFAULT_REPORT_DIR = os.path.join(HERE, "reports")

TARGET_SR = 16000
WIN = 15360      # 0.96 s
HOP = 7680       # 0.48 s
THRESHOLDS = (0.3, 0.5, 0.7)

PCM = 1
IEEE_FLOAT = 3
EXTENSIBLE = 0xFFFE

# Category name -> (primary class, related classes, short label).
# Names are verified against class_map.csv at runtime; anything missing is
# reported instead of silently ignored.
GROUPS = {
    "门铃": {
        "primary": "Doorbell",
        "related": ["Doorbell", "Ding-dong", "Chime", "Bell", "Wind chime",
                    "Church bell", "Tubular bells", "Telephone bell ringing"],
    },
    "敲门": {
        "primary": "Knock",
        "related": ["Knock", "Tap", "Door", "Thump, thud", "Thunk", "Slam",
                    "Hammer", "Wood"],
    },
    "烟雾报警": {
        "primary": "Smoke detector, smoke alarm",
        "related": ["Smoke detector, smoke alarm", "Alarm", "Fire alarm",
                    "Beep, bleep", "Buzzer", "Siren"],
    },
    "水壶": {
        "primary": "Steam whistle",
        "related": ["Steam whistle", "Whistle", "Boiling", "Steam", "Hiss"],
    },
    "婴儿哭声": {
        "primary": "Baby cry, infant cry",
        "related": ["Baby cry, infant cry", "Crying, sobbing", "Whimper",
                    "Wail, moan"],
    },
}

VERDICT_TEXT = {
    "A": "真正识别到目标类别",
    "B": "识别到语义相近类别（未命中主目标类）",
    "C": "完全没有识别到目标类别",
}


# --------------------------------------------------------------------------- #
# WAV reading - same semantics as yamnet_batch_eval.py
# --------------------------------------------------------------------------- #
class WavError(Exception):
    pass


def read_wav(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 12 or blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file")

    info = {"audio_format": None, "channels": None, "sample_rate": None,
            "bits": None, "data_bytes": 0}
    fmt = None
    raw = None
    pos = 12
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = int.from_bytes(blob[pos + 4:pos + 8], "little")
        body = blob[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            if len(body) < 16:
                raise WavError("truncated fmt chunk")
            fmt = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if fmt == EXTENSIBLE and len(body) >= 40:
                fmt = int.from_bytes(body[24:26], "little")
            info.update(audio_format=fmt, channels=channels,
                        sample_rate=rate, bits=bits)
        elif cid == b"data":
            raw = body
            info["data_bytes"] = len(body)
        pos += 8 + size + (size & 1)

    if fmt is None:
        raise WavError("no fmt chunk")
    if raw is None:
        raise WavError("no data chunk")
    if info["channels"] < 1:
        raise WavError("invalid channel count")

    bits, channels = info["bits"], info["channels"]
    if fmt == PCM and bits == 8:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif fmt == PCM and bits == 16:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif fmt == PCM and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b[: b.size - (b.size % 3)].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v).astype(np.float32)
        data = v / 8388608.0
    elif fmt == PCM and bits == 32:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif fmt == IEEE_FLOAT and bits == 32:
        data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif fmt == IEEE_FLOAT and bits == 64:
        data = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    else:
        raise WavError(f"unsupported encoding: format={fmt}, bits={bits}")

    if data.size % channels:
        data = data[: data.size - (data.size % channels)]
    data = data.reshape(-1, channels)
    if data.shape[0] == 0:
        raise WavError("zero audio frames")
    return data, info


def resample_fft(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x.astype(np.float32)
    n_in = x.size
    n_out = max(1, int(round(n_in * float(fs_out) / float(fs_in))))
    spec = np.fft.rfft(x.astype(np.float64))
    n_bins_in = spec.size
    n_bins_out = n_out // 2 + 1
    if n_bins_out <= n_bins_in:
        out = spec[:n_bins_out].copy()
        if n_out % 2 == 0 and n_bins_out <= n_bins_in:
            out[-1] = out[-1].real
    else:
        out = np.zeros(n_bins_out, dtype=np.complex128)
        out[:n_bins_in] = spec
    y = np.fft.irfft(out, n_out) * (float(n_out) / float(n_in))
    return y.astype(np.float32)


def to_mono_16k(data, info):
    mono = data.mean(axis=1) if info["channels"] > 1 else data[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if info["sample_rate"] != TARGET_SR:
        mono = resample_fft(mono, info["sample_rate"], TARGET_SR)
    return mono


# --------------------------------------------------------------------------- #
# Clipping check (read-only)
# --------------------------------------------------------------------------- #
def clipping_report(data, info):
    """Full-scale sample census, per channel. Nothing is modified."""
    bits = info["bits"]
    # value that means "pinned at the top/bottom of the integer range"
    full = 1.0 - (1.0 / (1 << (bits - 1))) if info["audio_format"] == PCM else 1.0
    thr = full - 1e-6

    per_ch = []
    total = 0
    longest_run = 0
    for c in range(data.shape[1]):
        x = np.abs(data[:, c])
        mask = x >= thr
        n = int(mask.sum())
        total += n
        run = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            run = max(run, cur)
        longest_run = max(longest_run, run)
        per_ch.append({"channel": c, "pinned_samples": n,
                       "peak": round(float(x.max()), 6),
                       "longest_consecutive_run": run})
    pct = (total / float(data.size) * 100.0) if data.size else 0.0
    return {
        "pinned_threshold": round(thr, 6),
        "pinned_samples_total": total,
        "pinned_percent": round(pct, 4),
        "longest_consecutive_run": longest_run,
        "per_channel": per_ch,
        "note": ("samples pinned at the integer full-scale limit; a consecutive "
                 "run >= 2 means flat-topped hard clipping"),
    }


def clipping_time_map(data, info, sr=TARGET_SR):
    """Which seconds of the file contain pinned samples (mono mix, per second)."""
    mono = np.abs(data).max(axis=1)
    bits = info["bits"]
    full = 1.0 - (1.0 / (1 << (bits - 1))) if info["audio_format"] == PCM else 1.0
    mask = mono >= (full - 1e-6)
    if not mask.any():
        return []
    sr_in = info["sample_rate"]
    idx = np.flatnonzero(mask)
    secs = {}
    for i in idx:
        s = int(i // sr_in)
        secs[s] = secs.get(s, 0) + 1
    return sorted(secs.items())


# --------------------------------------------------------------------------- #
# Frame utilities
# --------------------------------------------------------------------------- #
def frame_energy_db(x, n_frames):
    out = []
    for i in range(n_frames):
        s = i * HOP
        seg = x[s: s + WIN]
        if seg.size == 0:
            out.append(float("-inf"))
            continue
        r = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        out.append(20.0 * np.log10(r) if r > 0 else float("-inf"))
    return out


def counts_and_runs(scores, thresholds=THRESHOLDS):
    """Per-threshold frame count and longest consecutive run."""
    counts, runs = {}, {}
    above_all = {}
    for t in thresholds:
        mask = scores > t
        counts[str(t)] = int(mask.sum())
        run = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            run = max(run, cur)
        runs[str(t)] = run
        above_all[str(t)] = mask
    return counts, runs, above_all


def first_run_start(mask, min_len):
    cur = 0
    for i, v in enumerate(mask):
        cur = cur + 1 if v else 0
        if cur >= min_len:
            return i - min_len + 1
    return None


def segments(top1_idx, names, scores, t_of):
    """Merge consecutive frames with the same top-1 class into readable segments."""
    segs = []
    for i, k in enumerate(top1_idx):
        name = names[k]
        if segs and segs[-1]["class"] == name and i == segs[-1]["_last"] + 1:
            segs[-1]["_last"] = i
            segs[-1]["scores"].append(float(scores[i, k]))
        else:
            segs.append({"class": name, "_first": i, "_last": i,
                         "scores": [float(scores[i, k])]})
    for s in segs:
        s["start_s"] = round(t_of(s["_first"]), 2)
        s["end_s"] = round(t_of(s["_last"]) + HOP / float(TARGET_SR), 2)
        s["frames"] = s["_last"] - s["_first"] + 1
        s["score_mean"] = round(float(np.mean(s["scores"])), 4)
        s["score_max"] = round(float(np.max(s["scores"])), 4)
        s.pop("scores")
        s.pop("_first")
        s.pop("_last")
    return segs


# --------------------------------------------------------------------------- #
# Main diagnosis for one file
# --------------------------------------------------------------------------- #
def diagnose(path, cat, model, names, n2i, min_run):
    data, info = read_wav(path)
    x = to_mono_16k(data, info)
    t0 = time.perf_counter()
    out = model(x)
    scores = np.asarray(out[0] if isinstance(out, (list, tuple)) else out["scores"])
    infer_ms = (time.perf_counter() - t0) * 1000.0

    n_frames = scores.shape[0]
    energy = frame_energy_db(x, n_frames)
    t_of = lambda i: i * HOP / float(TARGET_SR)          # window start
    t_center = lambda i: (i * HOP + WIN / 2.0) / float(TARGET_SR)

    spec = GROUPS[cat]
    prim_name = spec["primary"]
    prim_idx = n2i.get(prim_name)

    # related classes resolved for this category
    rel = [(c, n2i[c]) for c in spec["related"] if c in n2i]
    rel_missing = [c for c in spec["related"] if c not in n2i]

    prim_scores = scores[:, prim_idx].astype(float) if prim_idx is not None else None

    # frame-wise max over related classes
    if rel:
        rel_mat = np.stack([scores[:, j] for _, j in rel], axis=1)
        rel_best_idx = rel_mat.argmax(axis=1)
        rel_max = rel_mat.max(axis=1)
        rel_max_class = [rel[k][0] for k in rel_best_idx]
    else:
        rel_max = np.zeros(n_frames)
        rel_max_class = ["-"] * n_frames

    # top-5 per frame
    order = np.argsort(scores, axis=1)[:, ::-1][:, :5]

    frames = []
    for i in range(n_frames):
        frames.append({
            "frame": i,
            "t_start_s": round(t_of(i), 3),
            "t_center_s": round(t_center(i), 3),
            "energy_dbfs": round(energy[i], 1) if np.isfinite(energy[i]) else None,
            "top5": [[names[int(k)], round(float(scores[i, k]), 4)] for k in order[i]],
            "primary_score": round(float(prim_scores[i]), 4) if prim_scores is not None else None,
            "related_max_score": round(float(rel_max[i]), 4),
            "related_max_class": rel_max_class[i],
            "related_scores": {c: round(float(scores[i, j]), 4) for c, j in rel},
        })

    # primary class statistics
    primary_stats = None
    if prim_scores is not None:
        c, r, mask = counts_and_runs(prim_scores)
        best_f = int(np.argmax(prim_scores))
        primary_stats = {
            "class": prim_name,
            "index": int(prim_idx),
            "best_score": round(float(prim_scores[best_f]), 4),
            "best_frame": best_f,
            "best_time_s": round(t_center(best_f), 3),
            "counts": c,
            "longest_run": r,
            "run_start_s": {t: (round(t_center(first_run_start(mask[t], r[t])), 2)
                                if r[t] and first_run_start(mask[t], r[t]) is not None else None)
                            for t in c},
        }

    # per-related-class statistics
    per_class = {}
    best_related = {"score": -1.0, "class": None, "frame": None, "time_s": None}
    for cname, j in rel:
        s = scores[:, j].astype(float)
        cc, rr, _ = counts_and_runs(s)
        bf = int(np.argmax(s))
        per_class[cname] = {
            "index": int(j),
            "best_score": round(float(s[bf]), 4),
            "best_frame": bf,
            "best_time_s": round(t_center(bf), 3),
            "counts": cc,
            "longest_run": rr,
        }
        if s[bf] > best_related["score"]:
            best_related = {"score": round(float(s[bf]), 4), "class": cname,
                            "frame": bf, "time_s": round(t_center(bf), 3)}

    rel_counts, rel_runs, _ = counts_and_runs(rel_max)

    # verdict: requires a consecutive run, never a single frame
    if primary_stats and primary_stats["longest_run"]["0.5"] >= min_run \
            and primary_stats["best_score"] >= 0.5:
        verdict = "A"
        evidence = (f"{prim_name} 连续 {primary_stats['longest_run']['0.5']} 帧 > 0.5，"
                    f"最高 {primary_stats['best_score']:.4f} @ {primary_stats['best_time_s']:.2f}s")
    else:
        hit = None
        for cname, j in rel:
            s = scores[:, j].astype(float)
            _, rr, _ = counts_and_runs(s)
            if rr["0.5"] >= min_run:
                bf = int(np.argmax(s))
                hit = (cname, rr["0.5"], float(s[bf]), t_center(bf))
                break
        if hit:
            verdict = "B"
            evidence = (f"{hit[0]} 连续 {hit[1]} 帧 > 0.5（语义相近类别），"
                        f"最高 {hit[2]:.4f} @ {hit[3]:.2f}s；主目标 {prim_name} 未达连续 "
                        f"{min_run} 帧")
        else:
            verdict = "C"
            evidence = (f"无任何目标相关类别连续 {min_run} 帧 > 0.5；"
                        f"主目标 {prim_name} 最高 "
                        f"{(primary_stats['best_score'] if primary_stats else float('nan')):.4f}")

    segs = segments(order[:, 0], names, scores, t_of)

    return {
        "file": os.path.relpath(path, HERE),
        "category": cat,
        "source_format": f"{info['sample_rate']} Hz, {info['channels']} ch, {info['bits']}-bit",
        "duration_s": round(x.size / float(TARGET_SR), 3),
        "frames": n_frames,
        "inference_ms": round(infer_ms, 1),
        "primary": primary_stats,
        "related": {
            "best": best_related,
            "counts": rel_counts,
            "longest_run": rel_runs,
            "per_class": per_class,
            "missing_in_vocab": rel_missing,
        },
        "verdict": verdict,
        "verdict_text": VERDICT_TEXT[verdict],
        "evidence": evidence,
        "segments": segs,
        "clipping": clipping_report(data, info),
        "clipping_by_second": clipping_time_map(data, info),
        "frames_detail": frames,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_json(results, meta, path):
    payload = {"meta": meta, "results": results}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_csv(results, path):
    # union of every related class across categories, stable order
    cols = []
    for cat in GROUPS:
        for c in GROUPS[cat]["related"]:
            if c not in cols:
                cols.append(c)

    header = (["file", "category", "frame", "t_start_s", "t_center_s", "energy_dbfs"]
              + [f"top{i+1}_class" for i in range(5)]
              + [f"top{i+1}_score" for i in range(5)]
              + ["primary_class", "primary_score", "related_max_class", "related_max_score"]
              + cols)

    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in results:
            for fr in r["frames_detail"]:
                row = [r["file"], r["category"], fr["frame"], fr["t_start_s"],
                       fr["t_center_s"], fr["energy_dbfs"]]
                for i in range(5):
                    row.append(fr["top5"][i][0] if i < len(fr["top5"]) else "")
                for i in range(5):
                    row.append(fr["top5"][i][1] if i < len(fr["top5"]) else "")
                row += [r["primary"]["class"] if r["primary"] else "",
                        fr["primary_score"], fr["related_max_class"], fr["related_max_score"]]
                for c in cols:
                    row.append(fr["related_scores"].get(c, ""))
                w.writerow(row)


def fmt_pct(a, b):
    return f"{a}/{b}"


def write_md(results, meta, path):
    L = []
    A = L.append
    A("# YAMNet 帧级诊断报告\n")
    A(f"- 模型：`{meta['model_dir']}`")
    A(f"- 类别表：`{meta['class_map']}`（{meta['n_classes']} 类，名称与索引均从该文件解析）")
    A(f"- 分帧：窗长 0.96 s / 步长 0.48 s（16 kHz），共 {meta['total_frames']} 帧")
    A(f"- 判定规则：主目标类**连续 ≥ {meta['min_run']} 帧** score > 0.5 记为 A；"
      f"否则任一相关类连续 ≥ {meta['min_run']} 帧 > 0.5 记为 B；否则 C")
    A("")
    A("> 所有 `score` 均为模型对该类的独立 sigmoid 输出，**不是准确率、不是校准置信度**，"
      "逐帧相加不等于 1。单帧超过阈值**不**视为识别成功，判定必须满足连续帧条件。")
    A("> 「连续帧 ≥ 3」是本诊断的判定参数，**不是报警阈值**。")
    A("> 下表「时间」为帧步进网格区间（每帧 0.48 s），实际判决窗口是该帧之后的 0.96 s。")
    A("> 完整逐帧数据（每帧时间、Top-5、各相关类分数）见 `frame_diagnosis.csv` 与 "
      "`frame_diagnosis.json`。")
    A("")

    # overall table
    A("## 一、总览\n")
    A("| 类别 | 文件 | 时长 | 帧数 | 主目标类 | 最高 score @ 时间 | 连续>0.5 | 判定 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in results:
        p = r["primary"]
        best = f"{p['best_score']:.4f} @ {p['best_time_s']:.2f}s" if p else "-"
        run = p["longest_run"]["0.5"] if p else 0
        A(f"| {r['category']} | {os.path.basename(r['file'])} | {r['duration_s']:.2f}s | "
          f"{r['frames']} | {p['class'] if p else '-'} | {best} | {run} 帧 | "
          f"**{r['verdict']}** {r['verdict_text']} |")
    A("")

    # per category
    A("## 二、分类别诊断\n")
    for cat in GROUPS:
        rs = [r for r in results if r["category"] == cat]
        if not rs:
            continue
        A(f"### {cat}（{len(rs)} 个文件）\n")
        for r in rs:
            p = r["primary"]
            A(f"**{os.path.basename(r['file'])}** — {r['source_format']}，"
              f"{r['duration_s']:.2f} s，{r['frames']} 帧，推理 {r['inference_ms']:.1f} ms")
            A("")
            if p:
                A(f"- 主目标 `{p['class']}`（索引 {p['index']}）：最高 **{p['best_score']:.4f}** "
                  f"@ {p['best_time_s']:.2f}s；"
                  f">0.3 {fmt_pct(p['counts']['0.3'], r['frames'])} 帧，"
                  f">0.5 {fmt_pct(p['counts']['0.5'], r['frames'])} 帧，"
                  f">0.7 {fmt_pct(p['counts']['0.7'], r['frames'])} 帧；"
                  f"最长连续 >0.5 = **{p['longest_run']['0.5']} 帧**")
            A(f"- 相关类最高：`{r['related']['best']['class']}` "
              f"**{r['related']['best']['score']:.4f}** @ {r['related']['best']['time_s']:.2f}s；"
              f"逐帧相关类最大值 >0.3 {fmt_pct(r['related']['counts']['0.3'], r['frames'])} 帧，"
              f">0.5 {fmt_pct(r['related']['counts']['0.5'], r['frames'])} 帧，"
              f">0.7 {fmt_pct(r['related']['counts']['0.7'], r['frames'])} 帧")
            A(f"- 判定：**{r['verdict']} — {r['verdict_text']}**")
            A(f"  - 依据：{r['evidence']}")
            # best frame top5
            if p:
                bf = next(f for f in r["frames_detail"] if f["frame"] == p["best_frame"])
                A(f"- 主目标最佳帧 #{bf['frame']}（{bf['t_center_s']:.2f}s，能量 "
                  f"{bf['energy_dbfs']} dBFS）Top-5：" +
                  " · ".join(f"{c} {s:.4f}" for c, s in bf["top5"]))
            A("")
            # segments (condensed timeline)
            segs = r["segments"]
            if segs:
                A(f"- Top-1 随时间变化（合并同类别连续帧，共 {len(segs)} 段"
                  f"{'，仅列前 12 段' if len(segs) > 12 else ''}）：")
                A("")
                A("  | 时间 | 帧数 | Top-1 | 均值 | 峰值 |")
                A("  |---|---|---|---|---|")
                for s in segs[:12]:
                    A(f"  | {s['start_s']:.2f}–{s['end_s']:.2f}s | {s['frames']} | "
                      f"{s['class']} | {s['score_mean']:.4f} | {s['score_max']:.4f} |")
                A("")
            missing = r["related"]["missing_in_vocab"]
            if missing:
                A(f"- ⚠️ 词表中不存在：{', '.join(missing)}")
                A("")

    # clipping
    A("## 三、削波检查（只读，未修改任何文件）\n")
    A("| 文件 | 满幅样本数 | 占比 | 最长连续 | 峰值 | 削波所在秒 |")
    A("|---|---|---|---|---|---|")
    for r in results:
        c = r["clipping"]
        secs = r["clipping_by_second"]
        sec_txt = "无" if not secs else ", ".join(f"{s}s({n})" for s, n in secs[:10])
        if len(secs) > 10:
            sec_txt += f" …共 {len(secs)} 秒"
        A(f"| {os.path.basename(r['file'])} | {c['pinned_samples_total']} | "
          f"{c['pinned_percent']:.4f}% | {c['longest_consecutive_run']} | "
          f"{c['per_channel'][0]['peak']:.6f} | {sec_txt} |")
    A("")
    A("说明：满幅样本指取值钉在整数满量程（16-bit 即 ±32767/−32768）的样本。"
      "连续 run ≥ 2 表示波形出现平顶，属于真削波；孤立 1–2 个样本多为量化端点，影响可忽略。")
    A("")
    A("**重要**：`yamnet_batch_eval.py` 自身的 `clipped_samples` 字段用 `|x| > 1.0` 判定，"
      "而 16-bit PCM 归一化后最大只有 0.99997，因此该计数器对 16-bit WAV **恒为 0**，"
      "不能作为无削波的证据。上表为直接读取整数样本重新统计的结果。")
    A("")

    # conclusions
    A("## 四、结论\n")
    A("### 4.1 分类别汇总\n")
    A("| 类别 | 文件数 | A 真正识别 | B 语义相近 | C 未识别 |")
    A("|---|---|---|---|---|")
    for cat in GROUPS:
        rs = [r for r in results if r["category"] == cat]
        if not rs:
            continue
        a = sum(1 for r in rs if r["verdict"] == "A")
        b = sum(1 for r in rs if r["verdict"] == "B")
        c = sum(1 for r in rs if r["verdict"] == "C")
        A(f"| {cat} | {len(rs)} | {a} | {b} | {c} |")
    A("")
    A("### 4.2 逐文件判定\n")
    by_v = {"A": [], "B": [], "C": []}
    for r in results:
        by_v[r["verdict"]].append(os.path.basename(r["file"]))
    A(f"- **A 真正识别到目标类别**（{len(by_v['A'])} 个）：" +
      (", ".join(by_v["A"]) if by_v["A"] else "无"))
    A(f"- **B 只识别到语义相近类别**（{len(by_v['B'])} 个）：" +
      (", ".join(by_v["B"]) if by_v["B"] else "无"))
    A(f"- **C 完全没识别到**（{len(by_v['C'])} 个）：" +
      (", ".join(by_v["C"]) if by_v["C"] else "无"))
    A("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def write_charts(results, path):
    """Dependency-free SVG chart sheet (no matplotlib needed)."""
    W, H, PAD_L, PAD_R, PAD_T, PAD_B = 720, 150, 46, 12, 16, 26
    panels = []
    for r in results:
        n = len(r["frames_detail"])
        if n == 0:
            continue
        dur = r["duration_s"]
        pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B
        pts_prim, pts_rel = [], []
        for f in r["frames_detail"]:
            t = f["t_center_s"]
            xx = PAD_L + (t / max(dur, 1e-6)) * pw
            if f["primary_score"] is not None:
                pts_prim.append(f"{xx:.1f},{PAD_T + (1 - f['primary_score']) * ph:.1f}")
            pts_rel.append(f"{xx:.1f},{PAD_T + (1 - f['related_max_score']) * ph:.1f}")
        grid = []
        for thr in (0.3, 0.5, 0.7):
            yy = PAD_T + (1 - thr) * ph
            grid.append(f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{PAD_L+pw}" y2="{yy:.1f}" '
                        f'stroke="#B4B2A9" stroke-width="0.5" stroke-dasharray="3 3"/>')
            grid.append(f'<text x="{PAD_L-6}" y="{yy+3:.1f}" font-size="9" fill="#888780" '
                        f'text-anchor="end">{thr}</text>')
        # energy strip
        ebar = []
        emin, emax = -70.0, 0.0
        for f in r["frames_detail"]:
            e = f["energy_dbfs"]
            if e is None:
                continue
            v = min(max((e - emin) / (emax - emin), 0), 1)
            t = f["t_center_s"]
            xx = PAD_L + (t / max(dur, 1e-6)) * pw
            h = v * 14
            ebar.append(f'<rect x="{xx-1.2:.1f}" y="{H-8-h:.1f}" width="2.4" height="{h:.1f}" '
                        f'fill="#D3D1C7"/>')
        color = {"A": "#0F6E56", "B": "#BA7517", "C": "#A32D2D"}[r["verdict"]]
        panels.append(f"""
<div style="margin: 18px 0 6px; font-size: 13px; font-weight: 500; color: var(--color-text-primary);">
  {os.path.basename(r["file"])} · {r["category"]} · {dur:.2f}s · 判定 {r["verdict"]}（{r["verdict_text"]}）
</div>
<svg viewBox="0 0 {W} {H}" width="100%" role="img"
     aria-label="{os.path.basename(r['file'])} 目标类分数随时间曲线，判定 {r['verdict']}">
  <rect x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" fill="none" stroke="#D3D1C7" stroke-width="0.5"/>
  {''.join(grid)}
  {''.join(ebar)}
  <polyline points="{' '.join(pts_rel)}" fill="none" stroke="#7F77DD" stroke-width="1.4"/>
  <polyline points="{' '.join(pts_prim)}" fill="none" stroke="{color}" stroke-width="1.8"/>
  <text x="{PAD_L}" y="{H-12}" font-size="9" fill="#888780">0s</text>
  <text x="{PAD_L+pw}" y="{H-12}" font-size="9" fill="#888780" text-anchor="end">{dur:.1f}s</text>
  <text x="{PAD_L}" y="{PAD_T-6}" font-size="9" fill="#5F5E5A">score</text>
</svg>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>YAMNet 帧级诊断曲线</title>
<style>
body{{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;margin:0;padding:24px;
background:#FBFAF6;color:#2C2C2A;}}
.wrap{{max-width:860px;margin:0 auto;}}
h1{{font-size:18px;font-weight:500;margin:0 0 6px;}}
p.note{{font-size:12px;color:#5F5E5A;line-height:1.7;margin:0 0 4px;}}
.legend{{display:flex;gap:18px;font-size:12px;color:#5F5E5A;margin:14px 0 4px;flex-wrap:wrap;}}
.sw{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px;}}
</style></head><body><div class="wrap">
<h1>YAMNet 帧级分数曲线</h1>
<p class="note">曲线为该类别的模型输出 score（独立 sigmoid），不是准确率、不是校准置信度。</p>
<div class="legend">
  <span><span class="sw" style="background:#0F6E56"></span>主目标类（判定 A 时深绿，B 琥珀，C 深红）</span>
  <span><span class="sw" style="background:#7F77DD"></span>相关类逐帧最大值</span>
  <span><span class="sw" style="background:#D3D1C7"></span>底部灰条 = 该帧能量（−70…0 dBFS）</span>
  <span>虚线 = 0.3 / 0.5 / 0.7</span>
</div>
{''.join(panels)}
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="YAMNet frame-level diagnosis")
    ap.add_argument("--dir", default=DEFAULT_AUDIO_DIR)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_REPORT_DIR)
    ap.add_argument("--min-run", type=int, default=3,
                    help="consecutive frames required for a positive verdict (diagnostic only)")
    args = ap.parse_args()

    import tensorflow as tf
    import tensorflow_hub as hub

    os.makedirs(args.out_dir, exist_ok=True)

    class_map = os.path.join(args.model_dir, "assets", "yamnet_class_map.csv")
    names = []
    with open(class_map, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() in ("index", ""):
                continue
            names.append(row[2])
    n2i = {n: i for i, n in enumerate(names)}

    print(f"[info] class map: {class_map}  ({len(names)} classes)")

    # resolve every configured class name against the real class map
    resolved = {}
    for cat, spec in GROUPS.items():
        entry = {"primary": None, "related": {}, "missing": []}
        for cname in [spec["primary"]] + spec["related"]:
            if cname in n2i:
                entry["related"][cname] = n2i[cname]
                if cname == spec["primary"]:
                    entry["primary"] = n2i[cname]
            else:
                ci = {k.lower(): v for k, v in n2i.items()}.get(cname.lower())
                if ci is not None:
                    real = names[ci]
                    entry["related"][real] = ci
                    if cname == spec["primary"]:
                        entry["primary"] = ci
                    entry["missing"].append(f"{cname} -> resolved case-insensitively as '{real}'")
                else:
                    entry["missing"].append(cname)
        resolved[cat] = entry
        print(f"[info] {cat}: primary index={entry['primary']}, "
              f"{len(entry['related'])} related resolved"
              + (f", MISSING={entry['missing']}" if entry["missing"] else ""))

    files = []
    for root, _, fns in os.walk(args.dir):
        for fn in sorted(fns):
            if fn.lower().endswith(".wav"):
                p = os.path.join(root, fn)
                cat = os.path.basename(root)
                if cat not in GROUPS:
                    print(f"[warn] skipping {p}: folder '{cat}' is not a known category")
                    continue
                files.append((cat, p))
    files.sort(key=lambda t: (list(GROUPS).index(t[0]), os.path.basename(t[1])))
    print(f"[info] {len(files)} wav file(s) under {args.dir}")

    t_load = time.perf_counter()
    model = hub.load(args.model_dir)
    print(f"[info] model loaded in {time.perf_counter()-t_load:.2f}s, "
          f"TF {tf.__version__}")

    results = []
    for cat, p in files:
        try:
            r = diagnose(p, cat, model, names, n2i, args.min_run)
            results.append(r)
            print(f"[ok] {cat}/{os.path.basename(p)}  frames={r['frames']}  "
                  f"verdict={r['verdict']}  primary_best="
                  f"{r['primary']['best_score'] if r['primary'] else 'n/a'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {cat}/{os.path.basename(p)}: {exc}")

    total_frames = sum(r["frames"] for r in results)
    meta = {
        "tool": "yamnet_frame_diagnosis.py",
        "model_dir": args.model_dir,
        "class_map": class_map,
        "n_classes": len(names),
        "target_sr": TARGET_SR,
        "window_samples": WIN,
        "hop_samples": HOP,
        "window_s": WIN / float(TARGET_SR),
        "hop_s": HOP / float(TARGET_SR),
        "thresholds": list(THRESHOLDS),
        "min_run": args.min_run,
        "min_run_note": "diagnostic criterion only, NOT an alarm threshold",
        "total_files": len(results),
        "total_frames": total_frames,
        "class_resolution": {c: {"primary": v["primary"],
                                 "related": v["related"],
                                 "notes": v["missing"]} for c, v in resolved.items()},
        "score_semantics": ("per-class independent sigmoid output; not accuracy, "
                            "not calibrated confidence, rows do not sum to 1"),
    }

    jp = os.path.join(args.out_dir, "frame_diagnosis.json")
    cp = os.path.join(args.out_dir, "frame_diagnosis.csv")
    mp = os.path.join(args.out_dir, "frame_diagnosis_summary.md")
    hp = os.path.join(args.out_dir, "frame_diagnosis_charts.html")

    write_json(results, meta, jp)
    write_csv(results, cp)
    write_md(results, meta, mp)
    write_charts(results, hp)

    print(f"[done] {jp}")
    print(f"[done] {cp}")
    print(f"[done] {mp}")
    print(f"[done] {hp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
