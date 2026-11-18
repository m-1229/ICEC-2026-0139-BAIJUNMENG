"""Batch WAV evaluation with the local YAMNet SavedModel.

For every .wav found in a folder:
  * inspect the source format (sample rate, channels, bit depth, duration, range)
  * convert to what YAMNet expects: mono, 16 kHz, float32, [-1, 1]
  * run inference once and report the Top-K AudioSet classes with scores
  * report per-file inference time and a run summary

Everything is standard library + numpy; the local model in ./models/yamnet is
reused, nothing is downloaded and no dependency is installed.

Wording note: the numbers printed are *model scores* (independent per-class
sigmoid outputs). They are not accuracy and not calibrated probabilities.

Usage:
    python yamnet_batch_eval.py --dir recordings
    python yamnet_batch_eval.py --dir recordings --recursive --json out.json --csv out.csv
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
DEFAULT_AUDIO_DIR = os.path.join(HERE, "samples")
TARGET_SR = 16000
TARGET_FRAMES = 15360        # 0.96 s analysis window at 16 kHz
TARGET_HOP = 7680            # 0.48 s hop
MIN_USEFUL_SECONDS = 0.96    # shorter than one window -> no real evidence
WARN_SILENT_DBFS = -50.0

PCM = 1
IEEE_FLOAT = 3
EXTENSIBLE = 0xFFFE


# --------------------------------------------------------------------------- #
# WAV reading (fmt 1 PCM 8/16/24/32 bit and fmt 3 float 32/64 bit)
# --------------------------------------------------------------------------- #
class WavError(Exception):
    pass


def read_wav(path):
    """Return (data[n, channels] float32 in [-1,1], info dict)."""
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
        elif cid == b"LIST" or cid == b"fact":
            pass
        pos += 8 + size + (size & 1)  # chunks are word aligned

    if fmt is None:
        raise WavError("no fmt chunk")
    if raw is None:
        raise WavError("no data chunk")
    if info["channels"] < 1:
        raise WavError(f"invalid channel count {info['channels']}")

    bits = info["bits"]
    channels = info["channels"]
    try:
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
    except ValueError as exc:
        raise WavError(f"cannot decode samples: {exc}") from exc

    if data.size % channels:
        data = data[: data.size - (data.size % channels)]
    data = data.reshape(-1, channels)
    if data.shape[0] == 0:
        raise WavError("zero audio frames")
    return data, info


def describe_format(info):
    kind = {PCM: "PCM", IEEE_FLOAT: "float"}.get(info["audio_format"],
                                                 str(info["audio_format"]))
    return f"{info['sample_rate']} Hz, {info['channels']} ch, {info['bits']}-bit {kind}"


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def resample_fft(x, fs_in, fs_out):
    """Spectral resampling (anti-aliased when downsampling), numpy only."""
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
            out[-1] = out[-1].real  # keep Nyquist bin real for even lengths
    else:
        out = np.zeros(n_bins_out, dtype=np.complex128)
        out[:n_bins_in] = spec
    y = np.fft.irfft(out, n_out) * (float(n_out) / float(n_in))
    return y.astype(np.float32)


def preprocess(data, info):
    """Mono 16 kHz float32 + a list of (level, text) notes."""
    notes = []
    channels = info["channels"]
    sr = info["sample_rate"]

    if channels > 1:
        mono = data.mean(axis=1)
        notes.append(("info", f"downmixed {channels} ch -> 1 ch (channel mean)"))
    else:
        mono = data[:, 0]

    finite = np.isfinite(mono)
    if not finite.all():
        bad = int((~finite).sum())
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        notes.append(("warn", f"{bad} non-finite samples replaced by 0"))

    peak_raw = float(np.max(np.abs(mono))) if mono.size else 0.0
    clipped = int(np.sum(np.abs(mono) > 1.0))
    if clipped:
        mono = np.clip(mono, -1.0, 1.0)
        notes.append(("warn", f"{clipped} samples exceeded [-1,1] and were clipped "
                              f"(source peak {peak_raw:.3f})"))

    if sr != TARGET_SR:
        mono = resample_fft(mono, sr, TARGET_SR)
        notes.append(("info", f"resampled {sr} Hz -> {TARGET_SR} Hz"))

    mono = np.ascontiguousarray(mono, dtype=np.float32)
    duration = mono.size / float(TARGET_SR)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    peak_db = 20.0 * np.log10(peak) if peak > 0 else float("-inf")
    dc = float(np.mean(mono)) if mono.size else 0.0

    if duration < MIN_USEFUL_SECONDS:
        notes.append(("warn", f"duration {duration:.2f} s < 0.96 s, too short for a "
                              f"full analysis window"))
    if peak == 0.0:
        notes.append(("warn", "signal is digital silence"))
    elif peak_db < WARN_SILENT_DBFS:
        notes.append(("warn", f"very quiet: peak {peak_db:.1f} dBFS"))
    if abs(dc) > 0.1:
        notes.append(("warn", f"large DC offset {dc:+.3f}"))

    stats = {
        "duration_s": round(duration, 3),
        "peak": round(peak, 4),
        "peak_dbfs": round(peak_db, 1) if np.isfinite(peak_db) else None,
        "rms": round(rms, 4),
        "dc_offset": round(dc, 4),
        "clipped_samples": clipped,
        "samples": int(mono.size),
    }
    return mono, stats, notes


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def load_class_names(class_map_path):
    names = []
    with open(class_map_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                names.append(row[2])
    return names


def find_wavs(folder, recursive):
    if recursive:
        found = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(".wav"):
                    found.append(os.path.join(root, name))
        return sorted(found)
    return sorted(
        os.path.join(folder, n) for n in os.listdir(folder)
        if n.lower().endswith(".wav") and os.path.isfile(os.path.join(folder, n))
    )


def evaluate_file(model, class_names, path, top_k):
    data, info = read_wav(path)
    waveform, stats, notes = preprocess(data, info)

    t0 = time.perf_counter()
    scores, embeddings, log_mel = model(waveform)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    scores = np.asarray(scores)
    embeddings = np.asarray(embeddings)
    log_mel = np.asarray(log_mel)

    mean_scores = scores.mean(axis=0)
    order = np.argsort(mean_scores)[::-1][:top_k]
    top = [{"rank": i + 1, "class": class_names[int(idx)],
            "score": round(float(mean_scores[int(idx)]), 4)}
           for i, idx in enumerate(order)]

    return {
        "file": path,
        "source_format": describe_format(info),
        "source_sample_rate": info["sample_rate"],
        "source_channels": info["channels"],
        "source_bits": info["bits"],
        "stats": stats,
        "notes": notes,
        "frames": int(scores.shape[0]),
        "shapes": {"scores": list(scores.shape),
                   "embeddings": list(embeddings.shape),
                   "log_mel": list(log_mel.shape)},
        "inference_ms": round(elapsed_ms, 1),
        "top": top,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_result(result, index, total, top_k):
    name = os.path.basename(result["file"])
    print(f"\n[{index}/{total}] {name}")
    print(f"  source : {result['source_format']}, {result['stats']['duration_s']:.2f} s")
    print(f"  prep   : 16 kHz mono float32, peak {result['stats']['peak']:.3f} "
          f"({result['stats']['peak_dbfs']} dBFS), dc {result['stats']['dc_offset']:+.3f}")
    for level, text in result["notes"]:
        print(f"  {level:<6} : {text}")
    print(f"  infer  : {result['frames']} frames, {result['inference_ms']:.1f} ms "
          f"(CPU)")
    print(f"  top-{top_k} :")
    for item in result["top"]:
        bar = "#" * int(round(item["score"] * 24))
        print(f"    {item['rank']}. {item['class']:<38} {item['score']:.4f}  {bar}")


def write_json(results, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"note": "scores are per-class model outputs, not calibrated "
                           "probabilities and not accuracy",
                   "results": results}, fh, ensure_ascii=False, indent=2)


def write_csv(results, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "duration_s", "source_format", "rank", "class",
                         "score", "frames", "inference_ms"])
        for result in results:
            for item in result["top"]:
                writer.writerow([result["file"], result["stats"]["duration_s"],
                                 result["source_format"], item["rank"],
                                 item["class"], item["score"], result["frames"],
                                 result["inference_ms"]])


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate WAV files with the local YAMNet model.")
    parser.add_argument("--dir", default=DEFAULT_AUDIO_DIR,
                        help="folder containing .wav files "
                             f"(default: {DEFAULT_AUDIO_DIR})")
    parser.add_argument("--recursive", action="store_true",
                        help="also scan sub-folders")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", default=None, help="write full results to JSON")
    parser.add_argument("--csv", default=None, help="write a flat table to CSV")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR folder not found: {args.dir}")
        return 2
    files = find_wavs(args.dir, args.recursive)
    if not files:
        print(f"ERROR no .wav files in {args.dir}"
              + ("" if args.recursive else " (try --recursive)"))
        return 2

    print("=" * 66)
    print("YAMNet batch evaluation")
    print("=" * 66)
    print(f"folder : {args.dir}  ({len(files)} wav file(s)"
          + (", recursive" if args.recursive else "") + ")")
    print(f"model  : {args.model_dir}")

    import tensorflow as tf
    import tensorflow_hub as hub

    t_load = time.perf_counter()
    model = hub.load(args.model_dir)
    print(f"loaded  : {time.perf_counter() - t_load:.2f} s (TF {tf.__version__})")
    class_names = load_class_names(model.class_map_path().numpy().decode("utf-8"))
    print(f"classes : {len(class_names)}")

    results, failures = [], []
    t_all = time.perf_counter()
    for i, path in enumerate(files, 1):
        try:
            result = evaluate_file(model, class_names, path, args.top_k)
        except (WavError, OSError) as exc:
            failures.append((path, str(exc)))
            print(f"\n[{i}/{len(files)}] {os.path.basename(path)}")
            print(f"  ERROR  : {exc}")
            continue
        results.append(result)
        print_result(result, i, len(files), args.top_k)

    total_s = time.perf_counter() - t_all
    print("\n" + "=" * 66)
    print(f"summary: {len(results)} ok, {len(failures)} failed, "
          f"{total_s:.1f} s total")
    if results:
        avg = sum(r["inference_ms"] for r in results) / len(results)
        print(f"         mean inference {avg:.1f} ms/file "
              f"({sum(r['stats']['duration_s'] for r in results):.1f} s audio)")
    if failures:
        for path, msg in failures:
            print(f"  failed: {os.path.basename(path)} - {msg}")
    print("note   : 'score' is the model output for that class "
          "(independent sigmoid),")
    print("         it is NOT accuracy and NOT a calibrated confidence.")

    if args.json:
        write_json(results, args.json)
        print(f"json   : {args.json}")
    if args.csv:
        write_csv(results, args.csv)
        print(f"csv    : {args.csv}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
