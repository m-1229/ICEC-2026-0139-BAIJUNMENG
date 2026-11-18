"""Doorbell / knock offline evaluation with the local YAMNet SavedModel.

What this script does
---------------------
For every sample in the fixed list below:

  * decode the WAV (format check, mono downmix, 16 kHz resample) using the
    helpers already present in ``yamnet_batch_eval.py`` -- nothing is re-implemented
  * run YAMNet once over the whole clip and keep the per-frame scores
  * build two "judgement curves", one per group:

        doorbell = max(Doorbell, Ding-dong, Chime, Bell, Wind chime)
        knock    = max(Knock, Door, Hammer)

  * smooth each curve with a 3-frame moving average
  * compare the smoothed curve against an EXPERIMENTAL threshold to record
    where (and whether) a trigger would have happened

Why a 3-frame mean instead of the clip mean
-------------------------------------------
The clip mean in ``yamnet_batch_eval.py`` averages every frame of the file,
including silence. A doorbell that rings once in a 12 s clip gets its single
0.98 evidence frame divided by 25 frames and ends up around 0.24. A short
moving average keeps the peak while still rejecting isolated single-frame
spikes. This is an aggregation change only -- the model, the window length and
the hop are untouched.

Wording discipline
------------------
The numbers printed here are per-class sigmoid outputs of the model. They are
NOT accuracy, NOT a calibrated probability, and NOT a validated threshold.
``THETA_DEFAULT = 0.45`` is a *recording* threshold for this experiment. It was
not fitted on a held-out set, so a "triggered" column here does not mean the
system works in a real room with music, TV and conversation.

``recordings/敲门/敲门.wav`` is kept as a true positive (the user listened to it
and confirmed it is a knock). YAMNet misses it because the impacts carry almost
no energy below 300 Hz. A low score is a property of the model, not evidence
about the label, so the label stays.

Usage
-----
    python yamnet_doorbell_eval.py
    python yamnet_doorbell_eval.py --theta 0.45
    python yamnet_doorbell_eval.py --dump 敲门          # frame-level dump
    python yamnet_doorbell_eval.py --json out.json --csv out.csv   (optional)
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
sys.path.insert(0, HERE)

from yamnet_batch_eval import (  # noqa: E402  (path set up above)
    WavError,
    describe_format,
    load_class_names,
    preprocess,
    read_wav,
)

DEFAULT_MODEL_DIR = os.path.join(HERE, "models", "yamnet")

WIN_S = 0.96          # YAMNet analysis window
HOP_S = 0.48          # YAMNet hop
SMOOTH_N = 3          # frames in the moving average
THETA_DEFAULT = 0.45  # experimental trigger level, see module docstring

GROUPS = {
    "doorbell": ["Doorbell", "Ding-dong", "Chime", "Bell", "Wind chime"],
    "knock": ["Knock", "Door", "Hammer"],
}

# role: "positive"          -> true event, expected to trigger
#       "positive_known_miss" -> true event, YAMNet is known to miss it
#       "control"           -> not a doorbell / knock, should stay quiet
SAMPLES = [
    ("recordings/门铃/doorbell_classic(1).wav", "doorbell", "positive",
     "电子门铃，含 2 个满幅削波样本"),
    ("recordings/门铃/doorbell_large.wav", "doorbell", "positive",
     "大钟式和弦门铃，余音长"),
    ("recordings/门铃/doorbell_mechanical.wav", "doorbell", "positive",
     "机械叮咚，仅 4.1 s / 8 帧，其中 4 帧静音"),
    ("recordings/门铃/门铃.wav", "doorbell", "positive",
     "混合铃音，命中 Chime / Wind chime"),
    ("recordings/敲门/knock_heavy.wav", "knock", "positive",
     "重敲，低频充足"),
    ("recordings/敲门/knock_light.wav", "knock", "positive",
     "轻敲，低频充足"),
    ("recordings/敲门/knock_medium.wav", "knock", "positive",
     "硬物撞击感，主要落在 Hammer"),
    ("recordings/敲门/敲门.wav", "knock", "positive_known_miss",
     "人工复听确认为真实敲门；YAMNet 已知漏检（低频缺失）。标签不因分数改变"),
    ("recordings/烟雾报警/smoke_alarm_long.wav", "control", "control",
     "对照：持续烟雾报警"),
    ("recordings/烟雾报警/smoke_alarm_standard.wav", "control", "control",
     "对照：脉冲烟雾报警"),
    ("recordings/水壶/kettle_whistle.wav", "control", "control",
     "对照：水壶笛音（历史误报源）"),
    ("recordings/婴儿哭声/baby_crying.wav", "control", "control",
     "对照：婴儿哭声"),
]

ROLE_TEXT = {
    "positive": "正样本",
    "positive_known_miss": "正样本（已知漏检）",
    "control": "对照",
}


# --------------------------------------------------------------------------- #
# judgement curves
# --------------------------------------------------------------------------- #
def group_curve(scores, class_idx):
    """Per-frame max over the classes of one group -> shape (frames,)."""
    return scores[:, class_idx].max(axis=1)


def moving_average(values, n):
    """Centred-free trailing mean over n consecutive frames (mode='valid')."""
    if values.size < n:
        return np.empty(0, dtype=np.float64)
    return np.convolve(values, np.ones(n, dtype=np.float64) / n, mode="valid")


def analyse(model, class_names, name_to_idx, path, theta):
    data, info = read_wav(path)
    waveform, stats, notes = preprocess(data, info)

    t0 = time.perf_counter()
    scores, _embeddings, _log_mel = model(waveform)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    scores = np.asarray(scores, dtype=np.float64)
    frames = int(scores.shape[0])

    result = {
        "file": path.replace(os.sep, "/"),
        "basename": os.path.basename(path),
        "source_format": describe_format(info),
        "duration_s": stats["duration_s"],
        "frames": frames,
        "stats": stats,
        "notes": notes,
        "inference_ms": round(elapsed_ms, 1),
        "groups": {},
    }

    for group_name, classes in GROUPS.items():
        col = [name_to_idx[c] for c in classes]
        raw = group_curve(scores, col)
        per_frame_best_class = scores[:, col].argmax(axis=1)

        k = int(np.argmax(raw))
        best_class = classes[int(per_frame_best_class[k])]

        # what the whole model says on that same frame, for context only
        top1_idx = int(np.argmax(scores[k]))
        top1 = (class_names[top1_idx], round(float(scores[k, top1_idx]), 4))

        entry = {
            "classes": classes,
            "best_raw": round(float(raw[k]), 4),
            "best_raw_class": best_class,
            "best_raw_frame": k,
            "best_raw_time": round(k * HOP_S, 2),
            "best_frame_top1": top1,
            "frames_over_theta": int((raw > theta).sum()),
        }

        smoothed = moving_average(raw, SMOOTH_N)
        if smoothed.size == 0:
            entry.update({
                "insufficient_frames": True,
                "best_smooth": None,
                "best_smooth_time": None,
                "trigger_windows": 0,
                "first_trigger_time": None,
                "first_trigger_window": None,
                "triggered": False,
            })
        else:
            j = int(np.argmax(smoothed))
            hits = np.where(smoothed > theta)[0]
            entry.update({
                "insufficient_frames": False,
                "best_smooth": round(float(smoothed[j]), 4),
                # decision time = end of the last frame in the 3-frame window
                "best_smooth_time": round((j + SMOOTH_N - 1) * HOP_S + WIN_S, 2),
                "trigger_windows": int(hits.size),
                "first_trigger_window": int(hits[0]) if hits.size else None,
                "first_trigger_time": (round((hits[0] + SMOOTH_N - 1) * HOP_S + WIN_S, 2)
                                       if hits.size else None),
                "triggered": bool(hits.size),
            })

        entry["_raw"] = raw
        entry["_smoothed"] = smoothed
        entry["_per_class"] = scores[:, col]   # frames x n_classes, for --dump
        result["groups"][group_name] = entry

    return result


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def print_file(result, index, total, theta):
    g = result["groups"]
    main = "knock" if g["knock"]["best_smooth"] is not None and \
        (g["knock"]["best_smooth"] or 0) >= (g["doorbell"]["best_smooth"] or 0) else "doorbell"

    print(f"\n[{index}/{total}] {result['basename']}   "
          f"({result['duration_s']} s, {result['frames']} 帧, "
          f"{result['inference_ms']:.0f} ms)")
    print(f"   源格式: {result['source_format']}")
    for level, text in result["notes"]:
        print(f"   {level:<5}: {text}")

    for name in ("doorbell", "knock"):
        e = g[name]
        mark = "*" if name == main else " "
        print(f"  {mark}[{name}] 组内类别: {', '.join(e['classes'])}")
        print(f"     最高原始帧分 : {e['best_raw']:.3f}  来自 {e['best_raw_class']}"
              f"  （帧 {e['best_raw_frame']} / {e['best_raw_time']:.2f} s）")
        print(f"     该帧全模型Top1: {e['best_frame_top1'][0]} {e['best_frame_top1'][1]:.3f}")
        if e["insufficient_frames"]:
            print(f"     3 帧均值     : 帧数不足（{result['frames']} < {SMOOTH_N}），无法构成窗口")
            print(f"     结论         : 无判定")
            continue
        print(f"     最高3帧均值  : {e['best_smooth']:.3f}"
              f"  （判定时刻 {e['best_smooth_time']:.2f} s）")
        print(f"     原始帧 >{theta:.2f} : {e['frames_over_theta']} 帧"
              f" | 判定窗口 >{theta:.2f} : {e['trigger_windows']} 个"
              f" | 首次触发 : "
              + (f"{e['first_trigger_time']:.2f} s" if e["triggered"] else "未触发"))
        print(f"     结论         : {'触发' if e['triggered'] else '未触发'}")


def print_table(results, theta):
    print("\n" + "=" * 118)
    print(f"汇总表（实验阈值 {theta:.2f}，仅记录触发情况，不是准确率）")
    print("=" * 118)
    head = (f"{'文件':<26}{'类别':<10}{'最高帧分':>9}{'来源类':>16}"
            f"{'最高3帧':>9}{'首次触发s':>11}{'触发窗口':>9}  结论")
    print(head)
    print("-" * 118)
    for r in results:
        for name in ("doorbell", "knock"):
            e = r["groups"][name]
            tag = f"{name}"
            first = "-" if e["first_trigger_time"] is None else f"{e['first_trigger_time']:.2f}"
            bs = "-" if e["best_smooth"] is None else f"{e['best_smooth']:.3f}"
            verdict = "触发" if e["triggered"] else "未触发"
            print(f"{r['basename']:<26}{tag:<10}{e['best_raw']:>9.3f}"
                  f"{e['best_raw_class']:>16}{bs:>9}{first:>11}"
                  f"{e['trigger_windows']:>9}  {verdict}")
    print("-" * 118)


def print_verdicts(registry, results, theta):
    print("\n" + "=" * 78)
    print("按样本角色核对")
    print("=" * 78)

    miss, false_alarm = [], []
    for (rel, category, role, note), r in zip(registry, results):
        group = "doorbell" if category == "doorbell" else "knock"
        if category == "control":
            fired = [n for n in ("doorbell", "knock") if r["groups"][n]["triggered"]]
            state = "有触发: " + ",".join(fired) if fired else "无触发"
            ok = not fired
            if fired:
                false_alarm.append(r["basename"])
        else:
            e = r["groups"][group]
            if e["triggered"]:
                state = f"触发 @ {e['first_trigger_time']:.2f} s"
                ok = True
            else:
                state = "未触发"
                ok = False
                if role == "positive_known_miss":
                    state += "（已登记为已知漏检）"
                else:
                    miss.append(r["basename"])
        print(f"  {'OK ' if ok else '!! '}{r['basename']:<26}{category:<10}"
              f"{ROLE_TEXT[role]:<18}{state}")
        if note:
            print(f"      备注: {note}")

    print("\n" + "-" * 78)
    print(f"门铃正样本触发 : {sum(1 for (x, c, ro, n), r in zip(registry, results) if c == 'doorbell' and r['groups']['doorbell']['triggered'])}"
          f" / {sum(1 for x, c, ro, n in registry if c == 'doorbell')}")
    print(f"敲门正样本触发 : {sum(1 for (x, c, ro, n), r in zip(registry, results) if c == 'knock' and r['groups']['knock']['triggered'])}"
          f" / {sum(1 for x, c, ro, n in registry if c == 'knock')}"
          f"（其中 敲门.wav 属已知漏检）")
    print(f"非预期漏检     : {len(miss)} 个" + (f" -> {', '.join(miss)}" if miss else ""))
    print(f"对照误触发     : {len(false_alarm)} 个"
          + (f" -> {', '.join(false_alarm)}" if false_alarm else ""))
    print("注意: 以上计数取自同一批文件，等于在同一样本上记录触发情况，")
    print("      不能当作准确率，也不能当作阈值已标定。")


def dump_frames(result, theta):
    door = result["groups"]["doorbell"]
    knock = result["groups"]["knock"]
    print("\n" + "=" * 112)
    print(f"逐帧明细: {result['basename']}   （实验阈值 {theta:.2f}）")
    print("=" * 112)
    head = (f"{'帧':>3}{'起s':>7}"
            + "".join(f"{c[:8]:>10}" for c in door["classes"])
            + f"{'门铃组':>9}{'3帧门铃':>9} | "
            + "".join(f"{c[:8]:>9}" for c in knock["classes"])
            + f"{'敲门组':>9}{'3帧敲门':>9}")
    print(head)
    print("-" * 112)
    ds, ks = door["_smoothed"], knock["_smoothed"]
    for i in range(result["frames"]):
        row = f"{i:>3}{i * HOP_S:>7.2f}"
        row += "".join(f"{v:>10.3f}" for v in door["_per_class"][i])
        row += f"{door['_raw'][i]:>9.3f}"
        wt = i - (SMOOTH_N - 1)
        row += f"{ds[wt]:>9.3f}" if 0 <= wt < ds.size else f"{'-':>9}"
        row += " | "
        row += "".join(f"{v:>9.3f}" for v in knock["_per_class"][i])
        row += f"{knock['_raw'][i]:>9.3f}"
        row += f"{ks[wt]:>9.3f}" if 0 <= wt < ks.size else f"{'-':>9}"
        print(row)
    print("-" * 112)
    print("说明: '3帧' 列是该帧作为窗口末帧时的滑动均值，'-' 表示窗口尚未形成。")


# --------------------------------------------------------------------------- #
# optional exports (off unless asked; this script adds no files by default)
# --------------------------------------------------------------------------- #
def export_json(results, registry, path, theta):
    payload = {"note": "scores are per-class sigmoid outputs, not accuracy and "
                       "not calibrated probabilities; threshold is experimental",
               "theta": theta, "smooth_frames": SMOOTH_N,
               "groups": GROUPS, "results": []}
    for (rel, category, role, note), r in zip(registry, results):
        item = {k: v for k, v in r.items()
                if not k.startswith("_") and k != "groups"}
        item.update({"sample_path": rel, "category": category, "role": role,
                     "role_note": note, "groups": {}})
        for name, e in r["groups"].items():
            clean = {k: v for k, v in e.items() if not k.startswith("_")}
            clean["raw_curve"] = [round(float(x), 4) for x in e["_raw"]]
            clean["smooth_curve"] = [round(float(x), 4) for x in e["_smoothed"]]
            item["groups"][name] = clean
        payload["results"].append(item)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def export_csv(results, registry, path, theta):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "category", "role", "group", "duration_s",
                         "frames", "best_raw", "best_raw_class",
                         "best_smooth", "first_trigger_s", "trigger_windows",
                         "frames_over_theta", "triggered", "theta"])
        for (rel, category, role, note), r in zip(registry, results):
            for name, e in r["groups"].items():
                writer.writerow([r["basename"], category, role, name,
                                 r["duration_s"], r["frames"], e["best_raw"],
                                 e["best_raw_class"], e["best_smooth"],
                                 e["first_trigger_time"], e["trigger_windows"],
                                 e["frames_over_theta"], e["triggered"], theta])


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Offline doorbell / knock evaluation with the local YAMNet model.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--theta", type=float, default=THETA_DEFAULT,
                        help=f"experimental trigger level (default {THETA_DEFAULT})")
    parser.add_argument("--dump", default=None,
                        help="print the frame-level table for files whose name "
                             "contains this substring")
    parser.add_argument("--json", default=None,
                        help="optional JSON export (off by default)")
    parser.add_argument("--csv", default=None,
                        help="optional CSV export (off by default)")
    args = parser.parse_args()

    import tensorflow as tf
    import tensorflow_hub as hub

    print("=" * 78)
    print("YAMNet 门铃 / 敲门离线评估")
    print("=" * 78)
    print(f"模型   : {args.model_dir}")
    print(f"规则   : 逐帧取组内最大 -> 最近 {SMOOTH_N} 帧滑动均值 -> > {args.theta:.2f} 触发")
    for name, classes in GROUPS.items():
        print(f"  {name:<8}: max({', '.join(classes)})")
    print("说明   : 分数是模型对单个类别的 sigmoid 输出，不是准确率，也不是校准概率；")
    print(f"         {args.theta:.2f} 仅为本次实验的记录阈值，未经标定。")

    missing = [p for p, _c, _r, _n in SAMPLES if not os.path.isfile(os.path.join(HERE, p))]
    if missing:
        print("\nERROR 以下样本文件不存在，停止执行（未做任何修改）:")
        for p in missing:
            print(f"  {p}")
        return 2

    t0 = time.perf_counter()
    model = hub.load(args.model_dir)
    class_names = load_class_names(model.class_map_path().numpy().decode("utf-8"))
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    absent = [c for cs in GROUPS.values() for c in cs if c not in name_to_idx]
    if absent:
        print(f"\nERROR 类别名在 class map 中不存在: {absent}")
        return 2
    print(f"类别   : {len(class_names)}；模型加载 {time.perf_counter() - t0:.2f} s "
          f"(TF {tf.__version__})")

    results, failed, pairs = [], [], []
    for i, sample in enumerate(SAMPLES, 1):
        rel, category, role, note = sample
        path = os.path.join(HERE, rel)
        try:
            result = analyse(model, class_names, name_to_idx, path, args.theta)
        except (WavError, OSError) as exc:
            failed.append((rel, str(exc)))
            print(f"\n[{i}/{len(SAMPLES)}] {rel}  ERROR: {exc}")
            continue
        results.append(result)
        pairs.append((sample, result))
        print_file(result, i, len(SAMPLES), args.theta)

    if not results:
        print("\n没有成功处理的文件，停止。")
        return 1

    ok_registry = [s for s, _r in pairs]

    print_table(results, args.theta)
    print_verdicts(ok_registry, results, args.theta)

    if args.dump:
        for r in results:
            if args.dump in r["basename"]:
                dump_frames(r, args.theta)

    if args.json:
        export_json(results, ok_registry, args.json, args.theta)
        print(f"\njson : {args.json}")
    if args.csv:
        export_csv(results, ok_registry, args.csv, args.theta)
        print(f"csv  : {args.csv}")

    if failed:
        print("\n失败文件:")
        for rel, msg in failed:
            print(f"  {rel} - {msg}")

    print("\n提醒: '触发' 只表示这条曲线在本次实验阈值下越过了 0.45，")
    print("      不代表真实环境下已经能可靠识别，误报面尚未测试。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
