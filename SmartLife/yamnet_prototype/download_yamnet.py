"""Download the official Google YAMNet model (Kaggle Models) into ./models/yamnet.

Official source (verified 2026-11):
    https://www.kaggle.com/api/v1/models/google/yamnet/TensorFlow2/yamnet/1/download
    -> HTTP 302 -> signed Google Cloud Storage URL (kaggle-models-data bucket)
    -> yamnet-tensorflow2-yamnet-v1.tar.gz, 14,242,921 bytes

The archive root is a plain TensorFlow SavedModel:
    assets/yamnet_class_map.csv, saved_model.pb, variables/*

Only the standard library is used, so no dependency needs to be installed.
Google domains are unreachable from some networks; pass --proxy if you need a
local VPN/HTTP proxy (it applies to this process only).

Usage:
    python download_yamnet.py
    python download_yamnet.py --proxy http://127.0.0.1:7890
"""

import argparse
import os
import sys
import tarfile
import urllib.request

DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/models/google/yamnet/TensorFlow2/yamnet/1/download"
)
EXPECTED_SIZE = 14_242_921
ARCHIVE_NAME = "yamnet-tensorflow2-yamnet-v1.tar.gz"
REQUIRED_FILES = (
    "saved_model.pb",
    "variables/variables.index",
    "assets/yamnet_class_map.csv",
)

HERE = os.path.dirname(os.path.abspath(__file__))


def build_opener(proxy):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def download(dest_path, proxy):
    opener = build_opener(proxy)
    print(f"[1/3] downloading {DOWNLOAD_URL}")
    if proxy:
        print(f"      via proxy {proxy}")
    with opener.open(DOWNLOAD_URL, timeout=120) as resp, open(dest_path, "wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            print(f"\r      {total / 1048576:.2f} MB", end="", flush=True)
    print()
    return total


def extract(archive_path, model_dir):
    print(f"[2/3] extracting to {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            tar.extractall(model_dir, filter="data")
        except TypeError:  # Python < 3.11.4 without extraction filters
            tar.extractall(model_dir)


def verify(model_dir):
    print("[3/3] verifying saved model layout")
    ok = True
    for rel in REQUIRED_FILES:
        path = os.path.join(model_dir, rel)
        if os.path.isfile(path):
            print(f"      OK   {rel:<40} {os.path.getsize(path):>10} bytes")
        else:
            print(f"      MISS {rel}")
            ok = False
    class_map = os.path.join(model_dir, "assets/yamnet_class_map.csv")
    if os.path.isfile(class_map):
        with open(class_map, "r", encoding="utf-8") as fh:
            rows = sum(1 for _ in fh) - 1
        print(f"      class_map rows: {rows} (expected 521)")
        ok = ok and rows == 521
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default=os.environ.get("YAMNET_PROXY"))
    parser.add_argument("--model-dir", default=os.path.join(HERE, "models", "yamnet"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive = os.path.join(os.path.dirname(args.model_dir), ARCHIVE_NAME)

    if os.path.isfile(os.path.join(args.model_dir, "saved_model.pb")) and not args.force:
        print(f"model already present in {args.model_dir} (use --force to re-download)")
        return 0 if verify(args.model_dir) else 1

    if os.path.isfile(archive) and os.path.getsize(archive) == EXPECTED_SIZE:
        print(f"archive already downloaded: {archive}")
    else:
        size = download(archive, args.proxy)
        if size != EXPECTED_SIZE:
            print(f"FAIL size mismatch: got {size}, expected {EXPECTED_SIZE}")
            return 1
        print(f"      size OK: {size} bytes")

    extract(archive, args.model_dir)
    if not verify(args.model_dir):
        print("FAIL incomplete model layout")
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
