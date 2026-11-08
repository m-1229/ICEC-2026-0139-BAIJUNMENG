#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 data/metrics.json 与原始样本序列注入模板，产出单文件 index.html（可离线双击打开）。"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
TPL = os.path.join(ROOT, "site", "template.html")
OUT = os.path.join(ROOT, "index.html")


def series_of(path):
    vals = []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        m = re.match(r"^(-?\d+(?:\.\d+)?),(\d{13})\s*$", line.strip())
        if m:
            vals.append(round(float(m.group(1)), 1))
    return vals


def main():
    payload = json.load(open(os.path.join(ROOT, "data", "metrics.json"), encoding="utf-8"))

    # 建立 (点位, 时段) -> 样本序列 索引（目录即元数据）
    idx = {}
    for per in ("早", "中", "晚"):
        for loc in payload["locations"]:
            f = os.path.join(RAW, per, f"{loc}.txt")
            if os.path.exists(f):
                idx[(loc, per)] = series_of(f)

    for r in payload["rows"]:
        s = idx.get((r["loc"], r["period"]))
        r["series"] = s if s else []

    html = open(TPL, encoding="utf-8").read()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = html.replace("/*__DATA__*/", data)
    open(OUT, "w", encoding="utf-8").write(html)
    kb = len(html.encode("utf-8")) / 1024
    print(f"written: {OUT}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
