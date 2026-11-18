"""Standalone YAMNet smoke test (no main-project coupling).

What it does:
  1. synthesises a 3 s mono 16 kHz WAV (1 kHz beeps + silence) with numpy only
  2. reads that WAV back from disk through the stdlib wave module
  3. loads the locally extracted YAMNet SavedModel with tensorflow_hub
  4. runs inference and validates shapes / numeric sanity of
     scores, embeddings and log_mel_spectrogram
  5. prints the Top-5 AudioSet classes and a PASS/FAIL verdict

Requirements already installed in .venv: tensorflow 2.15.1, tensorflow-hub 0.16.1,
tf-keras 2.15.1, numpy 1.26.4. No scipy / soundfile / librosa needed.

Usage:
    python yamnet_smoke_test.py
    python yamnet_smoke_test.py --audio path\\to\\clip.wav
"""

import argparse
import csv
import os
import sys
import time
import traceback
import wave

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

SAMPLE_RATE = 16000
DURATION_S = 3.0
MODEL_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "yamnet"
)
WAV_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "samples", "test_audio_16k.wav"
)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def synth_waveform():
    """Beep 0.4 s @ 1 kHz, 0.2 s silence, repeated; tail padded with silence."""
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    wave_data = np.zeros(n, dtype=np.float32)
    tone_len = int(0.4 * SAMPLE_RATE)
    gap_len = int(0.2 * SAMPLE_RATE)
    start = 0
    while start + tone_len <= n:
        seg_t = t[start : start + tone_len]
        wave_data[start : start + tone_len] = 0.6 * np.sin(
            2 * np.pi * 1000.0 * seg_t
        ).astype(np.float32)
        start += tone_len + gap_len
    # 5 ms fade in/out to avoid clicks
    fade = int(0.005 * SAMPLE_RATE)
    wave_data[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    wave_data[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return np.clip(wave_data, -1.0, 1.0)


def write_wav(path, waveform):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (waveform * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got sample width {width}")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return rate, data


def load_class_names(class_map_path):
    names = []
    with open(class_map_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header: index,mid,display_name
        for row in reader:
            if len(row) >= 3:
                names.append(row[2])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    parser.add_argument("--audio", default=None, help="use an existing 16 kHz mono wav")
    parser.add_argument("--wav-out", default=WAV_PATH_DEFAULT)
    args = parser.parse_args()

    print("=" * 62)
    print("YAMNet smoke test")
    print("=" * 62)

    print("\n[1] model directory")
    for rel in ("saved_model.pb", "variables/variables.index", "assets/yamnet_class_map.csv"):
        path = os.path.join(args.model_dir, rel)
        ok = os.path.isfile(path)
        check(f"file {rel}", ok, f"{os.path.getsize(path)} B" if ok else "missing")
    if not all(ok for _, ok, _ in results):
        print("\nRESULT: FAIL (incomplete model layout)")
        return 1

    print("\n[2] audio input")
    if args.audio:
        wav_path = args.audio
        print(f"  using existing file: {wav_path}")
    else:
        waveform = synth_waveform()
        write_wav(args.wav_out, waveform)
        wav_path = args.wav_out
        print(f"  synthesised -> {args.wav_out}")
    rate, wave_data = read_wav(wav_path)
    check("sample rate is 16 kHz", rate == SAMPLE_RATE, f"{rate} Hz")
    check("waveform mono & non-empty", wave_data.ndim == 1 and wave_data.size > 0, f"{wave_data.size} samples, {wave_data.size / rate:.2f} s")
    check("amplitude within [-1, 1]", float(np.max(np.abs(wave_data))) <= 1.0, f"peak {float(np.max(np.abs(wave_data))):.3f}")

    print("\n[3] load model")
    import tensorflow as tf
    import tensorflow_hub as hub

    t0 = time.time()
    model = hub.load(args.model_dir)
    load_s = time.time() - t0
    check("hub.load succeeded", True, f"{load_s:.2f} s, TF {tf.__version__}")

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = load_class_names(class_map_path)
    check("class map loaded", len(class_names) == 521, f"{len(class_names)} classes")

    print("\n[4] inference")
    t0 = time.time()
    scores, embeddings, log_mel = model(wave_data.astype(np.float32))
    infer_s = time.time() - t0
    scores = np.asarray(scores)
    embeddings = np.asarray(embeddings)
    log_mel = np.asarray(log_mel)
    print(f"  scores      {scores.shape} {scores.dtype}")
    print(f"  embeddings  {embeddings.shape} {embeddings.dtype}")
    print(f"  log_mel     {log_mel.shape} {log_mel.dtype}")
    print(f"  inference time {infer_s * 1000:.1f} ms")

    print("\n[5] shape validation")
    check("scores.ndim == 2", scores.ndim == 2, str(scores.shape))
    check("scores has 521 classes", scores.shape[1] == 521, f"got {scores.shape[1]}")
    check("embeddings.ndim == 2", embeddings.ndim == 2, str(embeddings.shape))
    check("embedding dim == 1024", embeddings.shape[1] == 1024, f"got {embeddings.shape[1]}")
    check("log_mel has 64 mel bins", log_mel.shape[1] == 64, f"got {log_mel.shape[1]}")
    check(
        "frame count consistent (scores == embeddings)",
        scores.shape[0] == embeddings.shape[0],
        f"{scores.shape[0]} frames",
    )
    # YAMNet frames the waveform with a 0.96 s window (15360 samples) and a
    # 0.48 s hop (7680 samples), and keeps the trailing partial window
    # (pad_end semantics) -> 1 + ceil((N - 15360) / 7680) frames.
    frame_length, frame_step = 15360, 7680
    span = max(0, wave_data.size - frame_length)
    expected_frames = 1 + (span + frame_step - 1) // frame_step
    check(
        "frame count matches 0.96s window / 0.48s hop framing",
        scores.shape[0] == expected_frames,
        f"expected {expected_frames}, got {scores.shape[0]}",
    )

    print("\n[6] numeric validation")
    check("all scores finite", bool(np.isfinite(scores).all()))
    check("scores within [0, 1]", float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0,
          f"min {float(scores.min()):.4f}, max {float(scores.max()):.4f}")
    row_sums = scores.sum(axis=1)
    print(
        f"  per-frame score sum: min {row_sums.min():.3f}, max {row_sums.max():.3f}"
        "  (521 independent sigmoid outputs, intentionally NOT normalised to 1)"
    )
    check("scores have a clear peak", float(scores.max()) > 0.1,
          f"max {float(scores.max()):.4f}")
    check("embeddings finite", bool(np.isfinite(embeddings).all()))
    check("embeddings not all zero", float(np.abs(embeddings).max()) > 0.0,
          f"max |emb| {float(np.abs(embeddings).max()):.4f}")
    check("log_mel finite", bool(np.isfinite(log_mel).all()))

    print("\n[7] Top-5 classes (mean over frames)")
    mean_scores = scores.mean(axis=0)
    top5 = np.argsort(mean_scores)[::-1][:5]
    for rank, idx in enumerate(top5, 1):
        print(f"  {rank}. {class_names[idx]:<40} {mean_scores[idx]:.4f}")

    failed = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 62)
    if failed:
        print(f"RESULT: FAIL - {len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"RESULT: PASS - all {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\nRESULT: FAIL — exception during run")
        traceback.print_exc()
        sys.exit(1)
