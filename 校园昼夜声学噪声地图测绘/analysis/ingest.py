#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补测数据入库工具
----------------
用法:
    python analysis/ingest.py <文件或压缩包> [更多文件...]
    python analysis/ingest.py --rebuild            # 仅重算 + 重建页面

支持:
    *.zip   微信收到的 Decibel X 导出包（内部一般只有一个 .txt）
    *.txt   Decibel X 直接导出的文本

自动从「文件名 / 压缩包名 / 内部文件名」识别 <时段><点位>，规范化写入
data/raw/<时段>/<点位>.txt，然后重算 data/acoustic_15groups.csv 与 index.html。

示例:
    python analysis/ingest.py 早主干道.zip 中主干道.zip 晚主干道.zip
"""

import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

PERIODS = ["早", "中", "晚"]
LOCS = ["图书馆", "食堂", "主干道", "操场", "实训室"]
ALIAS = {"饭堂": "食堂", "餐厅": "食堂", "运动场": "操场", "实训楼": "实训室", "主路": "主干道"}


def detect(name):
    """从任意字符串中识别 (时段, 点位)。"""
    s = name
    for a, b in ALIAS.items():
        s = s.replace(a, b)
    per = next((p for p in PERIODS if p in s), None)
    loc = next((l for l in LOCS if l in s), None)
    return per, loc


def looks_like_decibelx(data: bytes) -> bool:
    head = data[:400].decode("utf-8-sig", errors="ignore")
    return "Recorded date" in head and "Leq" in head


def ingest_one(path):
    """返回 [(per, loc, outpath)]"""
    out = []
    base = os.path.basename(path)
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if not n.lower().endswith(".txt"):
                    continue
                raw = z.read(n)
                if not looks_like_decibelx(raw):
                    print(f"  [skip] 非 Decibel X 记录: {n}")
                    continue
                per, loc = detect(base) if all(detect(base)) else (None, None)
                if not (per and loc):
                    per2, loc2 = detect(n)
                    per, loc = per or per2, loc or loc2
                if not (per and loc):
                    print(f"  [!] 无法识别时段/点位: {n}（文件名需含 早/中/晚 + 点位名）")
                    continue
                out.append(write(per, loc, raw))
    elif path.lower().endswith(".txt"):
        raw = open(path, "rb").read()
        if not looks_like_decibelx(raw):
            print(f"  [!] 不是 Decibel X 记录: {base}")
            return out
        per, loc = detect(base)
        if not (per and loc):
            print(f"  [!] 无法识别时段/点位: {base}")
            return out
        out.append(write(per, loc, raw))
    else:
        print(f"  [!] 不支持的格式: {base}")
    return out


def write(per, loc, raw):
    d = os.path.join(RAW, per)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f"{loc}.txt")
    if os.path.exists(dst):
        backup = dst + ".bak"
        shutil.move(dst, backup)
        print(f"  [i] 已备份原文件 -> data/raw/{per}/{loc}.txt.bak")
    open(dst, "wb").write(raw)
    print(f"  [ok] data/raw/{per}/{loc}.txt  ({len(raw)} bytes)")
    return per, loc, dst


def rebuild():
    for script in ("analyze.py", "build_site.py"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", script)],
                           capture_output=True, text=True, encoding="utf-8")
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr)
            raise SystemExit(f"{script} 执行失败")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args and "--rebuild" not in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    for p in args:
        print(f"→ {os.path.basename(p)}")
        if not os.path.exists(p):
            print("  [!] 文件不存在")
            continue
        ingest_one(p)
    print("\n重算指标并重建页面 …")
    rebuild()
    print("\n完成。打开 index.html 查看。")
