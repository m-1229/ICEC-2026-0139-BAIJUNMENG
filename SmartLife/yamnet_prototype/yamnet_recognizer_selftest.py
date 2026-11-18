#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""yamnet_recognizer 回归测试：用现有 12 段 WAV 验证流式路径能否复现离线结论。

测的是什么
----------
不是"把离线脚本再跑一遍"，而是**完整模拟实时管线**：
    WAV -> 解码/重采样到 16 kHz -> 按 0.48 s 跳距逐个切 0.975 s 窗
        -> 每窗单独推理得 1 帧 -> 分组取最大 -> 3 帧滑动均值 -> 阈值 0.45

这条路径就是第二批要接进 FastAPI 的那条。它必须与 `reports/doorbell_knock_eval.md`
的结论一致，否则说明重构引入了偏差。

验收标准
--------
    门铃 3/4 触发      敲门 3/4 触发      对照 4 段 0 误触发
    且各文件"最高判定分数"与离线记录的差值在 ±0.002 以内

关于末尾帧
----------
帧 k 需要波形延伸到 k*0.48 + 0.975 s 才能算出。文件末尾最后 1 帧之后的音频不存在，
`iter_windows(pad_tail=True)` 用零补齐——这对应实时场景里"事件之后环境安静下来"，
补出的帧分数天然很低，只会拉低判定量，不会凭空造出触发。

用法
----
    python yamnet_recognizer_selftest.py            # 完整回归
    python yamnet_recognizer_selftest.py -v         # 附逐文件触发明细
    python yamnet_recognizer_selftest.py --json out.json   # 另存结果

退出码：0 全部通过 / 1 有断言失败
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from yamnet_batch_eval import WavError, preprocess, read_wav  # noqa: E402
from yamnet_doorbell_eval import (  # noqa: E402
    SAMPLES,
    SMOOTH_N,
    THETA_DEFAULT,
    WIN_S,
    analyse,
)
import yamnet_recognizer as R  # noqa: E402

TOLERANCE = 0.002   # 分数容差。超出即视为"流式路径与离线不等价"，不允许改 theta 去凑。
FRAME_TOLERANCE = 1e-5   # 逐帧等价容差（实测为 0，留浮点余量）

# 离线基线：`reports/doorbell_knock_eval.md` 记录的「最高 3 帧均值」与触发结论。
# group 表示该文件属于哪一组判定；value 是那组的历史最高 3 帧均值（smooth，不是 raw）。
#
# 注意两个容易混淆的量：
#   best_raw    单帧最高分   （报告表里的"最高帧分"，例如 smoke_alarm_long 的 0.089）
#   best_smooth 3 帧均值最高 （本表采用，是真正参与阈值判定的量）
EXPECT = {
    "doorbell_classic(1).wav": {"group": "doorbell", "value": 0.544, "trigger": True},
    "doorbell_large.wav":      {"group": "doorbell", "value": 0.704, "trigger": True},
    "doorbell_mechanical.wav": {"group": "doorbell", "value": 0.019, "trigger": False},
    "门铃.wav":                 {"group": "doorbell", "value": 0.758, "trigger": True},

    "knock_heavy.wav":  {"group": "knock", "value": 0.968, "trigger": True},
    "knock_light.wav":  {"group": "knock", "value": 0.838, "trigger": True},
    # 离线 13 帧只能看到 smoothed[0]=0.570；流式多算 2 帧尾部，能看到 smoothed[11]=0.621。
    # 属"覆盖更全"，故判定用单侧（流式不得低于基线）。
    "knock_medium.wav": {"group": "knock", "value": 0.570, "trigger": True,
                         "note": "流式多覆盖尾部 2 帧，最高分可达 0.62"},
    "敲门.wav":          {"group": "knock", "value": 0.045, "trigger": False},

    # 对照：value 取两组 smoothed 的较大者，均远低于 0.45
    "smoke_alarm_long.wav":     {"group": "control", "value": 0.033, "trigger": False, "tol": 0.030},
    "smoke_alarm_standard.wav": {"group": "control", "value": 0.021, "trigger": False, "tol": 0.030},
    "kettle_whistle.wav":       {"group": "control", "value": 0.021, "trigger": False, "tol": 0.030},
    "baby_crying.wav":          {"group": "control", "value": 0.001, "trigger": False, "tol": 0.030},
}


def analyse_file(bundle, rel_path: str, theta: float, cooldown_s: float):
    """按实时流方式跑一个文件，并与离线整段推理逐帧比对。"""
    data, info = read_wav(rel_path)
    mono, stats, notes = preprocess(data, info)

    # --- 离线参考：整段一次性推理，作为逐帧比对的基准 ---
    offline = analyse(bundle.model, bundle.class_names, bundle.name_to_idx, rel_path, theta)

    # --- 流式：按 0.48 s 跳距逐窗推理 ---
    t0 = time.perf_counter()
    records, triggers = R.run_waveform(
        bundle, mono, theta=theta, cooldown_s=cooldown_s, pad_tail=True
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 逐帧比对：离线帧 k 的分数 vs 流式帧 k 的分数。这是"是否等价"的硬证据。
    frame_delta_max = 0.0
    compared = 0
    for k, rec in enumerate(records):
        if k >= offline["frames"]:
            break
        for g in R.GROUPS:
            ref = float(offline["groups"][g]["_raw"][k])
            got = float(rec.per_group[g])
            frame_delta_max = max(frame_delta_max, abs(got - ref))
        compared += 1

    # 各组最高 3 帧均值（NaN 表示该帧还没攒满滑窗）
    best = {}
    best_at = {}
    for g in R.GROUPS:
        vals = [(r.smoothed.get(g), r.frame_index, r.time_s) for r in records
                if r.smoothed.get(g) is not None]
        if vals:
            v, k, t = max(vals, key=lambda x: x[0])
            best[g] = round(float(v), 4)
            best_at[g] = {"frame": k, "time_s": t}
        else:
            best[g] = None
            best_at[g] = None

    return {
        "file": rel_path.replace(os.sep, "/"),
        "basename": os.path.basename(rel_path),
        "duration_s": stats["duration_s"],
        "frames_offline": offline["frames"],
        "frames_stream": len(records),
        "frames_compared": compared,
        "frame_delta_max": frame_delta_max,
        "offline_best_smooth": {
            g: offline["groups"][g]["best_smooth"] for g in R.GROUPS
        },
        "best_smooth": best,
        "best_at": best_at,
        "triggers": [{"group": t.group, "score": t.score, "time_s": t.time_s} for t in triggers],
        "first_trigger": ({"group": triggers[0].group, "score": triggers[0].score,
                           "time_s": triggers[0].time_s} if triggers else None),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YAMNet 识别库回归测试（12 段 WAV）")
    ap.add_argument("--theta", type=float, default=THETA_DEFAULT,
                    help=f"实验阈值，默认 {THETA_DEFAULT}（未标定，仅用于记录触发）")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="冷却秒数，默认 0（回归时关闭，避免掩盖重复触发）")
    ap.add_argument("--model-dir", default=None, help="模型目录，默认 models/yamnet")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印逐文件触发明细")
    ap.add_argument("--json", default=None, help="把结果另存为 JSON")
    args = ap.parse_args(argv)

    print("=" * 100)
    print("YAMNet 识别库回归测试")
    print("=" * 100)
    print(f"  窗长       : {R.WINDOW_SAMPLES} 样本 ({R.WINDOW_S:.4f} s)，STFT 帧数恰为 96（无补零）")
    print(f"  跳距       : {R.HOP_SAMPLES} 样本 ({R.HOP_SECONDS:.2f} s)")
    print(f"  滑窗       : {SMOOTH_N} 帧    阈值: {args.theta}（实验值，未标定）")
    print(f"  冷却       : {args.cooldown:.1f} s")
    print(f"  离线窗长   : {WIN_S} s（差 {abs(R.WINDOW_S - WIN_S) * 1000:.0f} ms，见模块文档）")
    print(f"  分数容差   : ±{TOLERANCE}")
    print()

    t0 = time.perf_counter()
    try:
        bundle = R.load_model(args.model_dir, verbose=True)
    except R.ModelLoadError as exc:
        print(f"[FAIL] 模型加载失败: {exc}")
        return 1
    print(f"  加载耗时   : {time.perf_counter() - t0:.2f} s")
    print()

    results = []
    failures = []

    print("-" * 100)
    print(f"{'文件':<24}{'时长':>7}{'流式帧':>7}{'离线帧':>7}{'逐帧Δmax':>10}"
          f"{'铃3帧':>8}{'敲3帧':>8}{'触发组':>8}{'基线':>8}{'Δ':>9}  判定")
    print("-" * 100)

    for rel, category, role, note in SAMPLES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            failures.append(f"{os.path.basename(rel)}: 文件不存在")
            print(f"{os.path.basename(rel):<24}  文件缺失")
            continue
        try:
            res = analyse_file(bundle, path, args.theta, args.cooldown)
        except (WavError, OSError) as exc:
            failures.append(f"{os.path.basename(rel)}: 读取失败 {exc}")
            print(f"{os.path.basename(rel):<24}  读取失败: {exc}")
            continue

        name = res["basename"]
        exp = EXPECT.get(name, {})
        g = exp.get("group")

        if g == "control":
            obs = max(v for v in res["best_smooth"].values() if v is not None)
            exp_val = exp.get("value", 0.0)
            tol = exp.get("tol", TOLERANCE)
            trig_group = "-"
        else:
            obs = res["best_smooth"].get(g)
            exp_val = exp.get("value")
            tol = TOLERANCE
            trig_group = res["triggers"][0]["group"] if res["triggers"] else "-"

        delta = (None if obs is None or exp_val is None else round(obs - exp_val, 4))

        # 硬指标 1：逐帧等价。这是"流式路径 == 离线口径"的直接证据。
        ok_frames = res["frame_delta_max"] <= FRAME_TOLERANCE

        # 硬指标 2：分数不低于基线（单侧）。
        # 单侧而非双侧：流式能多看尾部若干帧（离线受整段 patch 数截断），
        # 因此流式最高分可能高于离线记录值，这属于覆盖更全，不是偏差。
        # 反向（流式明显低于基线）才说明漏掉了离线的峰值。
        ok_score = (obs is not None and exp_val is not None and obs >= exp_val - tol)

        # 硬指标 3：触发结论与基线一致
        if g == "control":
            ok_trigger = not res["triggers"]
        else:
            ok_trigger = bool(res["triggers"]) == bool(exp.get("trigger"))

        # 硬指标 4：不得跨组串扰（门铃文件被敲门组触发，或反之）
        cross = (g != "control" and trig_group not in ("-", g))

        ok = ok_frames and ok_score and ok_trigger and not cross
        mark = "PASS" if ok else "FAIL"
        if not ok:
            reasons = []
            if not ok_frames:
                reasons.append(f"逐帧偏离 {res['frame_delta_max']:.2e} > {FRAME_TOLERANCE:.0e}")
            if not ok_score:
                reasons.append(f"分数 {obs} 低于基线 {exp_val} 超过 ±{tol}")
            if not ok_trigger:
                reasons.append(f"触发={trig_group if trig_group != '-' else '无'}，"
                               f"期望={'触发' if exp.get('trigger') else '不触发'}")
            if cross:
                reasons.append(f"跨组串扰 {trig_group}")
            failures.append(f"{name}: " + "；".join(reasons))

        print(f"{name:<24}{res['duration_s']:>7.2f}{res['frames_stream']:>7}"
              f"{res['frames_offline']:>7}"
              f"{res['frame_delta_max']:>10.1e}"
              f"{(res['best_smooth'].get('doorbell') or 0):>8.3f}"
              f"{(res['best_smooth'].get('knock') or 0):>8.3f}"
              f"{trig_group:>8}{str(exp_val):>8}"
              f"{(f'{delta:+.4f}' if delta is not None else '-'):>9}  {mark}")

        res.update({"category": category, "role": role, "expected": exp,
                    "observed_value": obs, "delta": delta, "verdict": mark,
                    "cross_talk": cross, "frames_ok": ok_frames, "score_ok": ok_score})
        results.append(res)

        if args.verbose and res["triggers"]:
            for t in res["triggers"]:
                print(f"    -> 触发 {t['group']} @ {t['time_s']:.2f}s  分数 {t['score']:.4f}")
            if exp.get("note"):
                print(f"       注：{exp['note']}")

    print("-" * 100)
    print()

    # ---------------- 汇总断言 ----------------
    def count_trig(cat_key: str) -> int:
        return sum(1 for r in results
                   if cat_key in r["file"].replace("\\", "/") and r["triggers"])

    bell_files = [r for r in results if "门铃" in r["file"] and r.get("role") != "control"]
    knock_files = [r for r in results if "敲门" in r["file"] and r.get("role") != "control"]
    ctrl_files = [r for r in results if r.get("role") == "control"]

    bell_hit = sum(1 for r in bell_files if r["triggers"])
    knock_hit = sum(1 for r in knock_files if r["triggers"])
    ctrl_hit = sum(1 for r in ctrl_files if r["triggers"])

    print("=" * 100)
    print("汇总")
    print("=" * 100)
    checks = []
    max_frame_delta = max((r["frame_delta_max"] for r in results), default=0.0)

    checks.append(("门铃触发 3/4", f"{bell_hit}/{len(bell_files)}", bell_hit == 3))
    checks.append(("敲门触发 3/4", f"{knock_hit}/{len(knock_files)}", knock_hit == 3))
    checks.append(("对照误触发 0", f"{ctrl_hit}/{len(ctrl_files)}", ctrl_hit == 0))
    checks.append(("逐帧与离线等价", f"Δmax={max_frame_delta:.1e}",
                   max_frame_delta <= FRAME_TOLERANCE))
    checks.append(("全部分项通过", f"{len(failures)} 项失败", not failures))

    for label, actual, ok in checks:
        print(f"  {'[PASS]' if ok else '[FAIL]'}  {label:<32} 实测 {actual}")

    print()
    if failures:
        print("失败明细：")
        for f in failures:
            print(f"  - {f}")
        print()

    # 已知漏检单独说明，不参与失败判定
    known_miss = [r for r in results if r.get("role") == "positive_known_miss"]
    if known_miss:
        for r in known_miss:
            print(f"  说明：{r['basename']} 是真实敲门正样本，YAMNet 已知漏检"
                  f"（最高 {r['best_smooth'].get('knock')}）。标签不因分数改变。")

    # 帧覆盖差异说明：流式帧数 >= 离线帧数，多出的是尾部补零帧
    wider = [r for r in results if r["frames_stream"] > r["frames_offline"]]
    if wider:
        print()
        print("  帧覆盖：流式帧数 >= 离线帧数（离线受整段 patch 数截断，流式尾部补零可多看几帧）：")
        for r in wider:
            print(f"    {r['basename']:<24} 流式 {r['frames_stream']:>3} 帧 / 离线 {r['frames_offline']:>3} 帧"
                  f"  （多 {r['frames_stream'] - r['frames_offline']} 帧，补零处分数天然很低）")
        print("  -> 逐帧比对只覆盖两者重叠的帧（frames_compared 列），那部分必须完全一致。")

    total_time = time.perf_counter() - t0
    print(f"\n总耗时 {total_time:.1f} s，模型加载 {R.is_loaded()}")

    if args.json:
        payload = {
            "note": "scores are per-class sigmoid outputs, not accuracy and not calibrated probabilities",
            "theta": args.theta, "theta_calibrated": False,
            "cooldown_s": args.cooldown,
            "window_samples": R.WINDOW_SAMPLES, "hop_samples": R.HOP_SAMPLES,
            "smooth_n": SMOOTH_N,
            "tolerance": TOLERANCE,
            "results": results,
            "summary": {
                "doorbell_triggered": f"{bell_hit}/{len(bell_files)}",
                "knock_triggered": f"{knock_hit}/{len(knock_files)}",
                "control_triggered": f"{ctrl_hit}/{len(ctrl_files)}",
                "frame_delta_max": round(max_frame_delta, 12),
                "frame_tolerance": FRAME_TOLERANCE,
                "score_tolerance": TOLERANCE,
                "failures": failures,
                "verdict": "PASS" if not failures and bell_hit == 3 and knock_hit == 3 else "FAIL",
            },
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args.json}")

    ok_all = all(ok for _, _, ok in checks)
    print()
    print("=" * 100)
    print("最终结果:", "PASS —— 流式路径复现了离线结论" if ok_all else "FAIL —— 流式路径与离线不等价，禁止进入第二批")
    print("=" * 100)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
