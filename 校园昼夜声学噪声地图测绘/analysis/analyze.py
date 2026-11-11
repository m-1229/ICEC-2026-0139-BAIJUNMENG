#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校园昼夜声学噪声地图测绘 —— 数据解析与指标计算
ICEC-2026-0139

输入 : _raw/*.txt        Decibel X 导出的原始记录（文件名格式：<时段><点位>__*.txt）
输出 : data/acoustic_15groups.csv     15 组声学数据表
       data/metrics.json              前端图表用的结构化数据
       docs/数据说明.md                数据来源与口径说明

指标口径
--------
L_Aeq   : 等效连续 A 声级，直接取 App 的 Avg (dB-A)
L_peak  : 峰值（Peak，声级计线性计权瞬时最大值），直接取 App 的 Peak
L_max   : 采样窗口内 1 s Leq 样本最大值
L_min   : 采样窗口内 1 s Leq 样本最小值
L10/L50/L90 : 统计声级，超过该声级的样本占 10% / 50% / 90%
σ       : 样本标准差，反映声环境稳定度
超标判定 : 按点位所属功能区昼夜限值比较（见 ZONES）
"""

import glob
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DATA = os.path.join(ROOT, "data")

PERIODS = ["早", "中", "晚"]
PERIOD_LABEL = {"早": "早间", "中": "午间", "晚": "晚间"}
PERIOD_WINDOW = {"早": "07:00–10:30", "中": "11:30–15:40", "晚": "18:30–21:10"}

# 点位 -> 功能区划 / 参考限值（昼 dB(A), 夜 dB(A)）
# 参考 GB 3096-2008《声环境质量标准》与 GB 50118-2010《民用建筑隔声设计规范》允许噪声级
ZONES = {
    "图书馆": {
        "code": "L",
        "zone": "1 类声环境功能区 · 图书馆阅览区（室内）",
        "limit_day": 50,
        "limit_night": 45,
        "ref": "GB 50118-2010 图书馆阅览室允许噪声级 ≤50 dB(A)；GB 3096-2008 1 类区夜间 45 dB(A)",
        "order": 1,
    },
    "食堂": {
        "code": "C",
        "zone": "2 类声环境功能区 · 食堂就餐区（室内）",
        "limit_day": 60,
        "limit_night": 50,
        "ref": "GB 50118-2010 餐厅/食堂允许噪声级 ≤60 dB(A)",
        "order": 2,
    },
    "主干道": {
        "code": "R",
        "zone": "4a 类声环境功能区 · 校园主干道（交通干线两侧）",
        "limit_day": 70,
        "limit_night": 55,
        "ref": "GB 3096-2008 4a 类区昼间 70 / 夜间 55 dB(A)",
        "order": 3,
    },
    "操场": {
        "code": "S",
        "zone": "1 类声环境功能区 · 运动场（室外）",
        "limit_day": 55,
        "limit_night": 45,
        "ref": "GB 3096-2008 1 类区昼间 55 / 夜间 45 dB(A)",
        "order": 4,
    },
    "实训室": {
        "code": "T",
        "zone": "3 类声环境功能区 · 实训室走廊（室内）",
        "limit_day": 65,
        "limit_night": 55,
        "ref": "GB 50118-2010 教学楼走廊允许噪声级；GB 3096-2008 3 类区夜间 55 dB(A)",
        "order": 5,
    },
}
LOC_ORDER = ["图书馆", "食堂", "主干道", "操场", "实训室"]

# 时段 -> 昼/夜
NIGHT_PERIODS = {"晚"}


def parse_file(path):
    """解析 Decibel X 文本导出。返回 (meta, samples)，samples 为 [(dB, ts)]。"""
    text = open(path, encoding="utf-8-sig", errors="replace").read()
    lines = [l.strip() for l in text.splitlines()]

    meta = {}
    for line in lines[:10]:
        m = re.match(r"^(Recorded date|Min \(dB-A\)|Max \(dB-A\)|Peak \(dB-A\)|Avg \(dB-A\)):\s*(.+)$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip()

    samples = []
    for line in lines:
        m = re.match(r"^(-?\d+(?:\.\d+)?),(\d{13})$", line)
        if m:
            samples.append((float(m.group(1)), int(m.group(2))))
    return meta, samples


def pctl(samples, p):
    """统计声级 Lx：升序后取第 p% 分位（超过该声级的样本占 (100-p)%）。"""
    if not samples:
        return float("nan")
    s = sorted(samples)
    idx = max(0, min(len(s) - 1, int(round((1 - p / 100.0) * (len(s) - 1)))))
    return s[idx]


def analyze():
    """扫描 data/raw/<时段>/<点位>.txt —— 目录即元数据，无需解析文件名。"""
    records = {}
    for per in PERIODS:
        for loc in LOC_ORDER:
            f = os.path.join(RAW, per, f"{loc}.txt")
            if not os.path.exists(f):
                continue
            meta, samples = parse_file(f)
            vals = [v for v, _ in samples]
            if not vals:
                print(f"[skip] 无样本: {f}")
                continue
            t0, t1 = samples[0][1], samples[-1][1]
            dur = (t1 - t0) / 1000.0
            dt = dur / max(1, len(vals) - 1)
            zone = ZONES[loc]
            is_night = per in NIGHT_PERIODS
            limit = zone["limit_night"] if is_night else zone["limit_day"]
            laeq = float(meta.get("Avg (dB-A)", statistics.fmean(vals)))
            lpeak = float(meta.get("Peak (dB-A)", max(vals)))
            records[(loc, per)] = {
                "id": "",
                "loc": loc,
                "period": per,
                "period_label": PERIOD_LABEL[per],
                "date": meta.get("Recorded date", ""),
                "file": f"data/raw/{per}/{loc}.txt",
                "zone": zone["zone"],
                "ref": zone["ref"],
                "limit": limit,
                "is_night": is_night,
                "duration_s": round(dur, 1),
                "samples": len(vals),
                "interval_ms": round(dt * 1000),
                "leaq": round(laeq, 1),
                "peak": round(lpeak, 1),
                "lmax": round(max(vals), 1),
                "lmin": round(min(vals), 1),
                "l10": round(pctl(vals, 10), 1),
                "l50": round(pctl(vals, 50), 1),
                "l90": round(pctl(vals, 90), 1),
                "sd": round(statistics.pstdev(vals), 2),
                "exceed_db": round(laeq - limit, 1),
                "exceed_peak": round(lpeak - limit, 1),
                "over": laeq > limit,
                "ratio": round(laeq / limit, 3),
            }

    # 15 组矩阵（主干道缺失时占位，便于后续补测自动填充）
    rows = []
    for loc in LOC_ORDER:
        for per in PERIODS:
            r = records.get((loc, per))
            if r:
                rows.append(r)
            else:
                zone = ZONES[loc]
                is_night = per in NIGHT_PERIODS
                rows.append({
                    "id": "",
                    "loc": loc, "period": per, "period_label": PERIOD_LABEL[per],
                    "date": "", "file": "",
                    "zone": zone["zone"], "ref": zone["ref"],
                    "limit": zone["limit_night"] if is_night else zone["limit_day"],
                    "is_night": is_night,
                    "duration_s": None, "samples": 0, "interval_ms": None,
                    "leaq": None, "peak": None, "lmax": None, "lmin": None,
                    "l10": None, "l50": None, "l90": None, "sd": None,
                    "exceed_db": None, "exceed_peak": None, "over": None,                     "ratio": None,
                    "pending": True,
                })
    # 统一编号 G01–G15
    for i, r in enumerate(rows, 1):
        r["no"] = f"G{i:02d}"
    # 现场干扰备注（测量人员现场观察，非仪器字段）
    QC_NOTE = {
        "G03": "含闭馆音乐：测量时馆内扩声系统持续播放，读数为音乐与背景的叠加值，不代表夜间本底",
        "G09": "含风噪：测量期间有风且点位开阔无遮挡，读数偏高约 6-10 dB，宜作上限参考",
    }
    for r in rows:
        r["note"] = QC_NOTE.get(r["no"], "")
    return rows


def to_csv(rows):
    cols = ["no", "loc", "period", "date", "duration_s", "samples", "interval_ms",
            "leaq", "peak", "lmax", "lmin", "l10", "l50", "l90", "sd",
            "limit", "exceed_db", "exceed_peak", "over", "ratio", "zone", "ref", "file", "note"]
    header = ["组号", "点位", "时段", "采样日期", "时长(s)", "样本数", "采样间隔(ms)",
              "Leq 均值 dB(A)", "Peak 峰值 dB(A)", "Lmax dB(A)", "Lmin dB(A)",
              "L10 dB(A)", "L50 dB(A)", "L90 dB(A)", "标准差 dB",
              "限值 dB(A)", "均值超标 dB", "峰值超标 dB", "是否超标", "比值",
              "功能区划", "限值依据", "源文件", "备注"]
    out = os.path.join(DATA, "acoustic_15groups.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if v is None:
                    v = "待补测"
                elif isinstance(v, bool):
                    v = "是" if v else "否"
                elif isinstance(v, str) and ("," in v or '"' in v):
                    v = '"' + v.replace('"', '""') + '"'
                vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print("written:", out)


def to_json(rows):
    payload = {
        "project": "校园昼夜声学噪声地图测绘",
        "code": "ICEC-2026-0139",
        "generated": "2026-11-08",
        "locations": LOC_ORDER,
        "periods": [{"key": p, "label": PERIOD_LABEL[p], "window": PERIOD_WINDOW[p]} for p in PERIODS],
        "zones": {k: {"zone": v["zone"], "limit_day": v["limit_day"],
                      "limit_night": v["limit_night"], "ref": v["ref"]} for k, v in ZONES.items()},
        "rows": rows,
    }
    out = os.path.join(DATA, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("written:", out)


if __name__ == "__main__":
    rows = analyze()
    to_csv(rows)
    to_json(rows)
    done = [r for r in rows if not r.get("pending")]
    print(f"\n完成 {len(done)}/15 组；缺失 {15 - len(done)} 组")
    for r in rows:
        if r.get("pending"):
            print(f"  待补测: {r['loc']} {r['period_label']}")
        else:
            flag = "超标" if r["over"] else "达标"
            print(f"  {r['no']:5s} {r['loc']:4s}{r['period_label']} "
                  f"Leq={r['leaq']:5.1f} Peak={r['peak']:5.1f} 限值={r['limit']:2d} "
                  f"[{flag}] {r['duration_s']:5.1f}s/{r['samples']}样本")
