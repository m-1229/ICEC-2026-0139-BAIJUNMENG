#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 3：真实物理麦克风联调（会真的打开麦克风，只在人工执行时使用）。

    yamnet_prototype\\.venv\\Scripts\\python.exe -B tools\\listen_stage3_real_mic.py --seconds 12

它做的事
--------
1. 枚举输入设备（不打开），打印候选与推荐项。
2. 按设备名称与 Host API 动态选择输入端点，默认优先 MME、DirectSound、
   WASAPI；`--device N` 可指定当前枚举中的索引。
3. 监听若干秒，每秒打印状态（帧数、触发、丢块、队列）**和实测电平**：
   `peak / peak_dbfs / 1s_max / sess_max / rms_dbfs / zero_blocks / silent_blocks`。
4. 逐帧打印模型分数（即使没有触发也打印）：
   `frame# / t / sm_doorbell / sm_knock / 是否触发`；结束时打印 `peak_smoothed`。
5. 实时打印声学影子门铃/敲门候选，同一 knock 簇只显示最新版本。
6. `POST /api/listen/stop`，确认流已关闭、worker 线程已退出，并打印开始/结束时间。
7. 前后比对数据库行数与音频文件，证明没有写库、没有录音。

判读要点
--------
* `level_peak≈1e-5..1e-3` 且 `zero_blocks=0` → 正常底噪（端点在工作）。
* `zero_blocks` 持续增长 → 数字零，端点不再供数。
* `sess_max`（会话最大峰值，**每秒上报一次**）能区分"真有声音"与"安静房间"。
* `sm_*` 是 3 帧均值，> theta 才触发；没触发也能看到它离阈值有多远。

不做：不写数据库、不保存音频、不跑模拟 WAV 冒充麦克风。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from listen_api_smoke import db_counts, free_port, get_status, http, start_server, stop_server  # noqa: E402
from listen_stage3_mic_smoke import audio_snapshot  # noqa: E402


def _level_line(mic: dict) -> str:
    """一行电平：最近块 / 1 秒窗口 / 会话最大 / RMS / 数字零与真实静音计数。"""
    return (
        f"lvl: peak={mic['level_peak']:.6f}({mic['level_peak_dbfs']:6.1f}dBFS) "
        f"1s_max={mic['level_peak_recent']:.6f} sess_max={mic['level_peak_max']:.6f} "
        f"rms={mic['level_rms_dbfs']:6.1f}dBFS "
        f"zero={mic['zero_blocks']} silent={mic['silent_blocks']} blocks={mic['level_blocks']}"
    )


def _score_str(value) -> str:
    return "   -   " if value is None else f"{value:.4f}"


def _print_frames(status: dict, last_frame: int) -> int:
    """打印新出现的逐帧分数（即使没有触发也打印）。"""
    for frame in status.get("frame_scores") or []:
        if frame["frame_index"] <= last_frame:
            continue
        last_frame = frame["frame_index"]
        flag = "  <== 触发" if frame["triggered"] else ""
        print(
            f"    frame#{frame['frame_index']:>3d} t={frame['time_s']:6.2f}s "
            f"sm_doorbell={_score_str(frame['doorbell'])} "
            f"sm_knock={_score_str(frame['knock'])}{flag}",
            flush=True,
        )
    return last_frame


def _candidate_key(candidate: dict) -> str:
    cluster_id = candidate.get("features", {}).get("cluster_id")
    if cluster_id:
        return str(cluster_id)
    return f"{candidate.get('candidate_type')}:{candidate.get('start_time_s')}"


def _print_acoustic_candidates(
    status: dict, last_signatures: dict[str, str]
) -> dict[str, str]:
    """打印新增或更新的影子候选；同一 knock 簇只显示最新版本。"""
    for candidate in status.get("acoustic", {}).get("candidates", []):
        key = _candidate_key(candidate)
        signature = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if last_signatures.get(key) == signature:
            continue
        last_signatures[key] = signature
        features = candidate.get("features", {})
        print(
            f"    -> 影子候选 {candidate['candidate_type']} "
            f"start={candidate['start_time_s']:.2f}s "
            f"last={candidate['time_s']:.2f}s "
            f"duration={candidate['duration_s']:.2f}s "
            f"rel={candidate['relative_db']:.1f}dB "
            f"impacts={features.get('impact_count', '-')} "
            f"cluster={features.get('cluster_id', '-')}",
            flush=True,
        )
        print(f"       {candidate['reason']}", flush=True)
    return last_signatures


def _ordered_device_candidates(
    devices: list[dict], requested: int | None, name_contains: str
) -> list[dict]:
    """按名称与 Host API 动态选择，绝不依赖固定设备索引。"""
    available = [d for d in devices if not d.get("excluded")]
    if requested is not None:
        matched = [d for d in devices if int(d["index"]) == int(requested)]
        return matched[:1]

    needle = name_contains.strip().lower()
    named = [
        d for d in available
        if needle and needle in str(d.get("name", "")).lower()
    ]
    pool = named or available

    def rank(device: dict) -> tuple[int, int]:
        hostapi = str(device.get("hostapi", "")).lower()
        for priority, part in enumerate(("mme", "directsound", "wasapi")):
            if part in hostapi:
                return priority, int(device["index"])
        return len(("mme", "directsound", "wasapi")), int(device["index"])

    return sorted(pool, key=rank)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--match",
        default="麦克风阵列",
        help="自动选择时使用的设备名称片段；留空则使用全部非排除设备。",
    )
    parser.add_argument(
        "--ready-delay",
        type=float,
        default=5.0,
        help="校准完成后留给用户准备声音的倒数秒数。",
    )
    args = parser.parse_args()

    before_db = db_counts()
    before_files = audio_snapshot()

    port = free_port()
    server, thread, base = start_server(port)
    print(f"服务已启动 {base}")
    exit_code = 0
    try:
        code, listing = http("GET", f"{base}/api/listen/devices")
        print(f"\n[设备] GET /api/listen/devices -> {code} available={listing.get('available')}")
        for d in listing.get("devices") or []:
            flag = "排除" if d["excluded"] else "候选"
            print(f"  idx={d['index']:<3d} {d['hostapi']:<22s} {d['default_samplerate']:>8.0f} Hz "
                  f"[{flag}] {d['name']}")
        target = listing.get("target") or {}
        print(f"  服务推荐：idx={target.get('index')} {target.get('name')} "
              f"({target.get('hostapi')}, {target.get('default_samplerate')} Hz)")

        candidates = _ordered_device_candidates(
            listing.get("devices") or [], args.device, args.match
        )
        if not candidates:
            print("  没有匹配的可用输入设备。")
            return 1
        print("  动态尝试顺序：")
        for device in candidates:
            print(f"    idx={device['index']} {device['hostapi']} {device['name']}")

        status = None
        for device in candidates:
            body = {"source": "microphone", "device": device["index"]}
            code, started = http("POST", f"{base}/api/listen/start", body)
            print(f"\n[启动] {device['hostapi']} idx={device['index']} -> {code}")
            if code != 202:
                print(f"  启动失败：{started}")
                http("POST", f"{base}/api/listen/stop")
                continue

            deadline = time.time() + 90
            candidate_status = started
            while time.time() < deadline:
                candidate_status = get_status(base)
                if candidate_status["state"] in ("listening", "error", "idle"):
                    break
                time.sleep(0.1)

            if candidate_status["state"] == "listening":
                status = candidate_status
                break
            print(
                f"  打开失败：state={candidate_status['state']} "
                f"err={candidate_status.get('last_error')}"
            )
            http("POST", f"{base}/api/listen/stop")
            time.sleep(0.2)

        if status is None or status["state"] != "listening":
            print("  所有动态候选设备均未能进入 listening。")
            return 1

        device = status["device"]
        print(f"\n[开流成功] state=listening")
        print(f"  device.index     = {device['index']}")
        print(f"  device.name      = {device['name']}")
        print(f"  device.hostapi   = {device['hostapi']}")
        print(f"  opened           = {device['opened']}")
        print(f"  实际参数         = {device['sample_rate']} Hz / "
              f"{device['channels']} ch / {device['dtype']}")
        print(f"  config           = {status['config']}")

        deadline = time.time() + 10
        while time.time() < deadline:
            status = get_status(base)
            acoustic = status["acoustic"]
            if acoustic.get("calibrated"):
                break
            if status["state"] != "listening":
                print(
                    f"  监听在校准期间异常停止：state={status['state']} "
                    f"err={status.get('last_error')}"
                )
                return 1
            time.sleep(0.05)
        else:
            print("  等待声学检测校准超时。")
            return 1

        print(
            f"\n[就绪] 声学校准完成，噪声底 "
            f"{status['acoustic']['noise_floor_dbfs']:.1f} dBFS"
        )
        for remaining in range(max(0, int(round(args.ready_delay))), 0, -1):
            print(f"  {remaining} 秒后开始正式监听计时……", flush=True)
            time.sleep(1.0)

        print(f"\n[监听 {args.seconds:.0f} s]")
        print(f"  服务端开始时间 = {status['started_at']}")
        print(f"  本机开始时间   = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  判据           = theta={status['config'].get('theta')} "
              f"smooth_n={status['config'].get('smooth_n')} "
              f"warmup={status['config'].get('warmup_frames')} "
              f"（分数是 3 帧均值，> theta 才触发）")
        start = time.time()
        last_seq = 0
        last_frame = -1
        last_candidate_signatures: dict[str, str] = {}
        while time.time() - start < args.seconds:
            time.sleep(1.0)
            status = get_status(base)
            frames = status["frames"]
            mic = status["mic"]
            print(f"  t={time.time() - start:5.1f}s state={status['state']:<9s} "
                  f"produced={frames['produced']:<4d} scored={frames['scored']:<4d} "
                  f"warmup={frames['warmup_dropped']} queue={mic['queue_depth']:<3d} "
                  f"drop={mic['dropped_blocks']:<3d} ovf={mic['overflow_events']:<3d} "
                  f"triggers={status['triggers']} "
                  f"acoustic_candidates={len(status['acoustic']['candidates'])}")
            print(f"         {_level_line(mic)}")
            last_frame = _print_frames(status, last_frame)
            for event in status["events"]:
                if event["seq"] > last_seq:
                    last_seq = event["seq"]
                    print(f"    -> 事件 #{event['seq']} {event['event_type']} "
                          f"score={event['score']:.4f} t={event['time_s']:.2f}s")
            last_candidate_signatures = _print_acoustic_candidates(
                status, last_candidate_signatures
            )

        code, stopped = http("POST", f"{base}/api/listen/stop")
        print(f"\n[停止] POST /api/listen/stop -> {code} state={stopped['state']} "
              f"reason={stopped['stop_reason']}")
        print(f"  本机结束时间   = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  服务端开始时间 = {stopped['started_at']}  "
              f"listening_s={stopped['listening_s']}  uptime_s={stopped['uptime_s']}")
        print(f"  device.opened = {stopped['device']['opened']}")
        time.sleep(0.4)
        alive = [t.name for t in __import__("threading").enumerate()
                 if t.name.startswith("listen-worker")]
        print(f"  残留 listen-worker = {alive}")
        last_frame = _print_frames(stopped, last_frame)
        mic = stopped["mic"]
        print(f"  本次会话事件数 = {len(stopped['events'])}  "
              f"帧数 = {stopped['frames']}  丢块 = {mic['dropped_blocks']}")
        last_candidate_signatures = _print_acoustic_candidates(
            stopped, last_candidate_signatures
        )
        print(
            "  影子候选汇总 = "
            + json.dumps(
                stopped["acoustic"]["candidates"],
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"  电平汇总：session_max_peak={mic['level_peak_max']:.6f}  "
              f"1s_max={mic['level_peak_recent']:.6f}  rms={mic['level_rms_dbfs']:.1f} dBFS  "
              f"数字零块={mic['zero_blocks']}  真实静音块={mic['silent_blocks']}  "
              f"统计块数={mic['level_blocks']}")
        print(f"  peak_smoothed = {stopped.get('peak_smoothed')}")
    finally:
        stop_server(server, thread)

    after_db = db_counts()
    after_files = audio_snapshot()
    added = sorted(set(after_files) - set(before_files))
    print(f"\n[核对] sound_events {before_db[0]} -> {after_db[0]}，"
          f"alert_settings {before_db[1]} -> {after_db[1]}，新增音频文件 {len(added)}")
    if before_db != after_db or added:
        exit_code = 1
    print("结论:", "PASS" if exit_code == 0 else "FAIL")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
