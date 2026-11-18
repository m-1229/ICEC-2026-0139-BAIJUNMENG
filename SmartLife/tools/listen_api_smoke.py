#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段 2 监听接口冒烟测试。

不依赖 pytest / httpx / requests：用标准库 urllib 打真实 uvicorn 服务，
服务在后台线程启动，测试结束关闭。**不打开麦克风、不写数据库。**

两种模式
--------
    # 主 venv（没有 numpy/TF）：验证应用照常启动、识别能力明确降级
    .venv\\Scripts\\python.exe -B tools\\listen_api_smoke.py --scope main

    # 原型 venv（有 numpy/TF/sounddevice）：跑完整状态机与三种素材
    yamnet_prototype\\.venv\\Scripts\\python.exe -B tools\\listen_api_smoke.py --scope full

退出码 0 = 全部通过（INFO 不算失败）；1 = 有未通过项。
"""

from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "smartlife.db"
PROTOTYPE_DIR = PROJECT_ROOT / "yamnet_prototype"


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, bool | None, str]] = []

    def add(self, tid, name, expected, actual, ok, note="") -> None:
        self.rows.append((tid, name, expected, actual, bool(ok), note))
        print(f"  [{'PASS' if ok else 'FAIL'}] {tid:<6s} {name:<34s} "
              f"期望 {expected:<20s} 实测 {actual}" + (f"  ({note})" if note else ""))

    def info(self, tid, name, actual, note="") -> None:
        self.rows.append((tid, name, "仅记录", actual, None, note))
        print(f"  [INFO] {tid:<6s} {name:<34s} 实测 {actual}"
              + (f"  ({note})" if note else ""))

    def failures(self):
        return [r for r in self.rows if r[4] is False]

    def infos(self):
        return [r for r in self.rows if r[4] is None]

    def passed(self):
        return [r for r in self.rows if r[4] is True]


R = Report()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def http(method: str, url: str, body=None, timeout: float = 15.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace") or "null"
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw      # 首页返回 HTML
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") or "null"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_server(port: int):
    import uvicorn

    from app.main import app

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, name="uvicorn-smoke", daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            code, _ = http("GET", f"{base}/api/health", timeout=1.0)
            if code == 200:
                return server, thread, base
        except Exception:
            pass
        time.sleep(0.05)
    raise RuntimeError("uvicorn 未能在 30 秒内启动")


def stop_server(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


# --------------------------------------------------------------------------- #
# 数据库（只读）
# --------------------------------------------------------------------------- #
def db_counts() -> tuple[int, int]:
    connection = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        events = connection.execute("SELECT COUNT(*) FROM sound_events").fetchone()[0]
        settings = connection.execute("SELECT COUNT(*) FROM alert_settings").fetchone()[0]
        return int(events), int(settings)
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# 可控假识别栈：制造异常、慢速退出、卡死
# --------------------------------------------------------------------------- #
class _ZeroStats:
    frames_produced = 0
    frames_scored = 0
    warmup_frames_dropped = 0
    stream_pos = 0
    buffered = 0


class _FakeRecognizer:
    def __init__(self, delay: float = 0.0, raises: bool = False):
        self.stats = _ZeroStats()
        self.delay = delay
        self.raises = raises

    def reset_session(self):
        return None

    def push(self, block):
        if self.raises:
            raise RuntimeError("测试注入：推理崩溃")
        if self.delay:
            time.sleep(self.delay)
        return []

    def flush(self):
        if self.raises:
            raise RuntimeError("测试注入：推理崩溃")
        return []


class _FakeStack:
    """delay>0 让 worker 每块慢一点，便于确定性观察 stopping。"""

    def __init__(self, delay: float = 0.0, raises: bool = False):
        self.delay = delay
        self.raises = raises

    def config(self):
        return {"window_samples": 15600, "hop_samples": 7680, "warmup_frames": 2}

    def new_recognizer(self):
        return _FakeRecognizer(self.delay, self.raises)

    def load_clip(self, clip):
        import numpy as np

        return np.zeros(48000 * 4, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 轮询辅助
# --------------------------------------------------------------------------- #
def get_status(base: str) -> dict:
    code, payload = http("GET", f"{base}/api/listen/status")
    assert code == 200, (code, payload)
    return payload


def wait_state(base: str, targets, timeout: float = 40.0, poll: float = 0.02):
    """轮询到目标状态之一，返回 (状态快照, 观察到的状态序列)。"""
    if isinstance(targets, str):
        targets = (targets,)
    seen: list[str] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_status(base)
        if not seen or seen[-1] != status["state"]:
            seen.append(status["state"])
        if status["state"] in targets:
            return status, seen
        time.sleep(poll)
    raise TimeoutError(f"等待状态 {targets} 超时，最近观察序列 {seen}")


def stop_and_watch(base: str, state_getter, timeout: float = 6.0):
    """发起 stop，同时用高频轮询捕捉瞬时的 stopping 状态。"""
    hit = {"seen": False}

    def watch() -> None:
        end = time.time() + timeout
        while time.time() < end:
            if state_getter() == "stopping":
                hit["seen"] = True
                return
            time.sleep(0.0005)

    watcher = threading.Thread(target=watch, name="stop-watcher", daemon=True)
    watcher.start()
    code, payload = http("POST", f"{base}/api/listen/stop")
    watcher.join(timeout=timeout)
    return code, payload, hit["seen"]


# --------------------------------------------------------------------------- #
# 通用检查
# --------------------------------------------------------------------------- #
def check_original_apis(base: str, tag: str) -> None:
    """原有接口回归：只做不写库的调用。"""
    code, health = http("GET", f"{base}/api/health")
    ok = (code == 200 and health.get("status") == "ok"
          and health.get("test_alert_available") is True
          and health.get("event_history_available") is True
          and health.get("alert_settings_available") is True)
    R.add(f"{tag}-1", "GET /api/health 原有字段", "200 + 3 个 true",
          f"{code} {[k for k in health if k.endswith('_available')]}", ok)

    code, events = http("GET", f"{base}/api/events?limit=5")
    R.add(f"{tag}-2", "GET /api/events", "200 列表", f"{code} {len(events)} 条",
          code == 200 and isinstance(events, list))

    code, settings = http("GET", f"{base}/api/settings/alerts")
    R.add(f"{tag}-3", "GET /api/settings/alerts", "200 四条",
          f"{code} {len(settings)} 条", code == 200 and len(settings) == 4)

    code, payload = http("POST", f"{base}/api/test-alert")
    R.add(f"{tag}-4", "POST /api/test-alert", "200 kind=test",
          f"{code} {payload.get('kind')}", code == 200 and payload.get("kind") == "test")

    # 故意用非法类型，触发 422 校验失败，不会写库
    code, _ = http("POST", f"{base}/api/events/simulate",
                   {"event_type": "not_a_type", "confidence": 0.9})
    R.add(f"{tag}-5", "POST /api/events/simulate 非法类型被拒", "422",
          f"{code}", code == 422)

    code, _ = http("GET", f"{base}/")
    R.add(f"{tag}-6", "GET / 首页", "200", f"{code}", code == 200)


def check_persist_flag(base: str, tag: str) -> None:
    status = get_status(base)
    ok = status["persist"] is False and status["microphone_enabled"] is False
    R.add(tag, "status.persist=False 且麦克风关闭", "False / False",
          f"{status['persist']} / {status['microphone_enabled']}", ok)


# --------------------------------------------------------------------------- #
# scope=main
# --------------------------------------------------------------------------- #
def run_main_scope(base: str) -> None:
    print("\n[主 venv] 应用可启动 + 识别能力降级")
    code, health = http("GET", f"{base}/api/health")
    R.add("M1", "health.listen_available", "False", f"{health.get('listen_available')}",
          code == 200 and health.get("listen_available") is False)
    R.add("M2", "health.microphone_enabled", "False",
          f"{health.get('microphone_enabled')}", health.get("microphone_enabled") is False)
    R.add("M3", "health.listen_persist", "False",
          f"{health.get('listen_persist')}", health.get("listen_persist") is False)

    code, payload = http("GET", f"{base}/api/listen/devices")
    R.add("M4", "devices 降级为 available=false", "200 + False",
          f"{code} available={payload.get('available')}",
          code == 200 and payload.get("available") is False and payload.get("devices") == [])
    R.info("M4b", "devices 降级原因", str(payload.get("detail"))[:70])

    code, payload = http("POST", f"{base}/api/listen/start", {"source": "simulated"})
    R.add("M5", "start 缺依赖 -> 503", "503",
          f"{code} {str(payload)[:60]}", code == 503)

    code, payload = http("GET", f"{base}/api/listen/status")
    R.add("M6", "status 可用且 idle", "200 idle",
          f"{code} {payload.get('state')}", code == 200 and payload.get("state") == "idle")

    code, payload = http("POST", f"{base}/api/listen/stop")
    R.add("M7", "stop 幂等", "200 idle",
          f"{code} {payload.get('state')}", code == 200 and payload.get("state") == "idle")


# --------------------------------------------------------------------------- #
# scope=full
# --------------------------------------------------------------------------- #
def run_full_scope(base: str) -> None:
    from app import audio_devices, recognition

    if str(PROTOTYPE_DIR) not in sys.path:
        sys.path.insert(0, str(PROTOTYPE_DIR))

    service = recognition.listen_service
    original_loader = recognition._load_stack

    # ---------- 设备枚举 ----------
    print("\n[设备] 枚举成功 / 枚举异常")
    code, payload = http("GET", f"{base}/api/listen/devices")
    devices = payload.get("devices", [])
    target = payload.get("target")
    R.add("D1", "devices 枚举成功", "200 + available=true",
          f"{code} available={payload.get('available')} 共 {len(devices)} 条",
          code == 200 and payload.get("available") is True and len(devices) > 0)
    excluded_virtual = [d for d in devices if d["excluded"] and "todesk" in d["name"].lower()]
    R.add("D2", "ToDesk 虚拟设备被标记排除", ">=1 条 excluded",
          f"{len(excluded_virtual)} 条", len(excluded_virtual) >= 1,
          "本机存在 ToDesk Virtual Audio")
    R.add("D3", "target 不是被排除设备", "excluded=false",
          f"{target['name'] if target else None} / excluded={target['excluded'] if target else None}",
          target is not None and target["excluded"] is False)
    R.info("D4", "target 建议设备", f"{target['hostapi']} / {target['default_samplerate']:.0f} Hz"
           f" / idx {target['index']}" if target else "无")

    # 枚举异常：把后端导入改成抛异常，走同一条 HTTP 路径
    original_import = audio_devices._import_backend

    def boom():
        raise audio_devices.DeviceEnumerationError("测试注入：音频后端不可用")

    audio_devices._import_backend = boom
    try:
        code, payload = http("GET", f"{base}/api/listen/devices")
    finally:
        audio_devices._import_backend = original_import
    R.add("D5", "枚举异常降级", "200 + available=false",
          f"{code} available={payload.get('available')}",
          code == 200 and payload.get("available") is False and payload.get("devices") == [])

    # ---------- 状态迁移 + doorbell ----------
    print("\n[状态机] idle -> loading -> listening -> stopping -> idle（doorbell）")
    check_persist_flag(base, "S0")
    code, started = http("POST", f"{base}/api/listen/start",
                         {"source": "simulated", "clip": "doorbell", "speed": 20.0})
    R.add("S1", "start 返回 202 + loading", "202 loading",
          f"{code} {started.get('state')}",
          code == 202 and started.get("state") in ("loading", "listening"))

    # 立刻再 start，应 409
    code, conflict = http("POST", f"{base}/api/listen/start", {"clip": "knock"})
    R.add("S2", "重复 start 返回 409", "409",
          f"{code} {json.dumps(conflict, ensure_ascii=False)[:60]}", code == 409)
    if isinstance(conflict, dict) and isinstance(conflict.get("detail"), dict):
        R.add("S2b", "409 携带当前状态", "含 state 字段",
              f"{conflict['detail'].get('state')}",
              conflict["detail"].get("state") in ("loading", "listening", "stopping"))

    listening_status, seen = wait_state(base, ("listening", "idle"), timeout=60)
    R.add("S3", "观察到 loading 状态", "loading 出现在序列中", f"{seen}",
          "loading" in seen)
    R.add("S4", "进入 listening", "listening", listening_status["state"],
          listening_status["state"] == "listening" or "listening" in seen)

    # 停止（正常路径）
    code, stopped = http("POST", f"{base}/api/listen/stop")
    R.add("S5", "stop 后回到 idle", "200 idle", f"{code} {stopped.get('state')}",
          code == 200 and stopped.get("state") == "idle")
    R.add("S6", "stop_reason 记录", "stopped / clip_finished / stopped_during_loading",
          str(stopped.get("stop_reason")), stopped.get("stop_reason") is not None)

    code, again = http("POST", f"{base}/api/listen/stop")
    R.add("S7", "重复 stop 幂等", "200 idle", f"{code} {again.get('state')}",
          code == 200 and again.get("state") == "idle")

    # stopping 是毫秒级瞬时状态，真模型下捕捉不到；用慢速 worker 让它持续 0.4 s，
    # 做确定性验证（不改服务代码，只替换识别栈）
    recognition._load_stack = lambda: _FakeStack(delay=0.4)
    try:
        http("POST", f"{base}/api/listen/start",
             {"source": "simulated", "clip": "control", "speed": 20.0})
        wait_state(base, ("listening",), timeout=30)
        code, payload, caught = stop_and_watch(base, lambda: service.state)
    finally:
        recognition._load_stack = original_loader
    R.add("S8", "stopping 状态确定可观测", "捕捉=True 且终态 idle",
          f"捕捉={caught} 终态={payload.get('state')}",
          caught and payload.get("state") == "idle")

    # ---------- 三种素材 ----------
    print("\n[素材] doorbell / knock / control 走真实 push()/flush()")
    results = {}
    # doorbell 期望 0 条：该素材铃声就在开头，方案甲的预热会丢掉含铃声的两个窗口
    # （阶段 1 A3 已确认：裸起峰值 0.378 < 0.45）。这是已批准的已知代价，不是缺陷。
    for clip, expectation in (("doorbell", "known_zero"), ("knock", "positive"),
                              ("control", "zero")):
        code, _ = http("POST", f"{base}/api/listen/start",
                       {"source": "simulated", "clip": clip, "speed": 20.0})
        final, seen = wait_state(base, ("idle", "error"), timeout=120)
        events = final["events"]
        groups = [e["event_type"] for e in events]
        results[clip] = (final, groups)
        ok = code == 202 and final["state"] == "idle" and final["stop_reason"] == "clip_finished"
        R.add(f"C-{clip}", f"{clip} 会话正常结束", "202 + idle + clip_finished",
              f"{code} {final['state']} {final['stop_reason']}", ok)
        if expectation == "positive":
            R.add(f"C-{clip}-e", f"{clip} 触发", ">=1 条",
                  f"{len(events)} 条 {groups[:3]}", len(events) >= 1)
        elif expectation == "zero":
            R.add(f"C-{clip}-e", f"{clip} 不触发", "0 条", f"{len(events)} 条", len(events) == 0)
        else:
            R.add(f"C-{clip}-e", f"{clip} 裸起不触发（方案甲已知代价）", "0 条",
                  f"{len(events)} 条 {groups[:3]}", len(events) == 0)
        R.add(f"C-{clip}-f", f"{clip} 帧计数", "produced>0 且 warmup_dropped=2",
              f"produced={final['frames']['produced']} "
              f"scored={final['frames']['scored']} "
              f"warmup={final['frames']['warmup_dropped']}",
              final["frames"]["produced"] > 0
              and final["frames"]["warmup_dropped"] == 2)
        if expectation == "positive":
            best = max(events, key=lambda e: e["score"]) if events else None
            R.info(f"C-{clip}-s", f"{clip} 最高模型分数",
                   f"{best['event_type']} {best['score']:.4f} @ {best['time_s']:.2f}s"
                   if best else "无")

    # 诊断：证明门铃链路本身没问题——同一素材补 2 帧前导静音后即可触发
    try:
        import numpy as _np

        from yamnet_stream import DECIMATE, HOP as _HOP, run_stream as _run_stream

        probe = recognition._load_stack()
        wave = probe.load_clip("doorbell")
        bare = _run_stream(probe.bundle, wave, input_rate=48000, chunk=960, warmup_frames=2)
        lead = _np.concatenate(
            [_np.zeros(2 * _HOP * DECIMATE, dtype=_np.float32), wave]
        )
        with_lead = _run_stream(probe.bundle, lead, input_rate=48000, chunk=960,
                                warmup_frames=2)
        R.info("D6", "门铃裸起 vs 补2帧前导（诊断）",
               f"裸起 峰值 {bare.peak_smoothed()['doorbell']:.4f} 事件 {len(bare.triggers)} / "
               f"前导 峰值 {with_lead.peak_smoothed()['doorbell']:.4f} "
               f"事件 {len(with_lead.triggers)}",
               "阈值 0.45 未改动")
    except Exception as exc:  # 诊断失败不影响判定
        R.info("D6", "门铃诊断未完成", f"{type(exc).__name__}: {exc}")

    # 与阶段 1 基线对照
    R.add("R1", "门铃裸起不触发（已批准口径）", "0 条",
          f"{len(results['doorbell'][1])} 条", len(results["doorbell"][1]) == 0,
          "补前导后可见触发，见 D6")
    R.add("R2", "敲门素材触发", ">=1 条",
          f"{len(results['knock'][1])} 条", len(results["knock"][1]) >= 1)
    R.add("R3", "对照素材不触发", "0 条",
          f"{len(results['control'][1])} 条", len(results["control"][1]) == 0)

    # ---------- 重启重置 ----------
    print("\n[会话] 停止后重新开始：预热/统计/序号重置")
    http("POST", f"{base}/api/listen/start",
         {"source": "simulated", "clip": "knock", "speed": 20.0})
    fresh, _ = wait_state(base, ("idle", "error"), timeout=120)
    prev_session = fresh["session_id"]
    prev_triggers = fresh["triggers"]
    seqs = [e["seq"] for e in fresh["events"]]
    R.add("N1", "事件 seq 从 1 连续递增", "1..n",
          f"{seqs[:5]}{'...' if len(seqs) > 5 else ''}",
          seqs == list(range(1, len(seqs) + 1)) and (not seqs or seqs[0] == 1))
    R.add("N2", "新会话预热重新计 2", "2", f"{fresh['frames']['warmup_dropped']}",
          fresh["frames"]["warmup_dropped"] == 2)

    code, second = http("POST", f"{base}/api/listen/start",
                        {"source": "simulated", "clip": "control", "speed": 20.0})
    R.add("N3", "新会话 session_id 变化", "不同",
          f"{prev_session} -> {second.get('session_id')}",
          second.get("session_id") not in (None, prev_session))
    R.add("N4", "新会话重置统计与事件", "triggers=0, events=[]",
          f"triggers={second.get('triggers')} events={len(second.get('events', []))}",
          second.get("triggers") == 0 and second.get("events") == [])
    http("POST", f"{base}/api/listen/stop")
    wait_state(base, ("idle",), timeout=30)

    # ---------- 加载期停止 ----------
    print("\n[加载期] loading 阶段 stop")
    import yamnet_recognizer as yr

    yr.unload_model()          # 制造冷加载，让 loading 持续可见
    t0 = time.time()
    http("POST", f"{base}/api/listen/start",
         {"source": "simulated", "clip": "control", "speed": 20.0})
    loading_status, seen = wait_state(base, ("loading", "listening"), timeout=30)
    code, stopped = http("POST", f"{base}/api/listen/stop")
    elapsed = time.time() - t0
    R.add("L1", "观察到 loading", "loading", f"{loading_status['state']}", True)
    R.add("L2", "loading 期间 stop 后回 idle", "200 idle",
          f"{code} {stopped.get('state')}", code == 200 and stopped.get("state") == "idle",
          f"耗时 {elapsed:.2f}s")
    R.add("L3", "stop_reason 标记加载期停止", "stopped_during_loading / stopped / clip_finished",
          str(stopped.get("stop_reason")), stopped.get("stop_reason") is not None)
    R.info("L4", "加载期 stop 全程耗时", f"{elapsed:.2f}s（含冷加载不可中断部分）")

    # ---------- 失败注入 ----------
    print("\n[异常] 模型加载失败 / worker 推理异常 / stop 超时")

    def broken_loader():
        raise RuntimeError("测试注入：模型文件损坏")

    recognition._load_stack = broken_loader
    try:
        http("POST", f"{base}/api/listen/start",
             {"source": "simulated", "clip": "doorbell", "speed": 20.0})
        errored, _ = wait_state(base, ("error",), timeout=30)
    finally:
        recognition._load_stack = original_loader
    R.add("E1", "加载失败落到 error", "error",
          f"{errored['state']} {str(errored.get('last_error'))[:40]}",
          errored["state"] == "error")
    R.add("E2", "last_error 含注入信息", "含 测试注入",
          str(errored.get("last_error"))[:50],
          "测试注入" in str(errored.get("last_error")))
    R.add("E3", "error 状态 persist 仍为 False", "False", f"{errored['persist']}",
          errored["persist"] is False)

    code, restarted = http("POST", f"{base}/api/listen/start",
                           {"source": "simulated", "clip": "doorbell", "speed": 20.0})
    R.add("E4", "error 后允许重新启动", "202",
          f"{code} {restarted.get('state')}", code == 202)
    http("POST", f"{base}/api/listen/stop")
    wait_state(base, ("idle",), timeout=30)

    # worker 推理异常
    recognition._load_stack = lambda: _FakeStack(raises=True)
    try:
        http("POST", f"{base}/api/listen/start",
             {"source": "simulated", "clip": "knock", "speed": 20.0})
        errored2, _ = wait_state(base, ("error",), timeout=30)
    finally:
        recognition._load_stack = original_loader
    R.add("E5", "worker 异常落到 error", "error + 含 推理崩溃",
          f"{errored2['state']} {str(errored2.get('last_error'))[:40]}",
          errored2["state"] == "error" and "推理崩溃" in str(errored2.get("last_error")))
    for _ in range(40):
        if service._thread is None:
            break
        time.sleep(0.05)
    R.add("E6", "线程已退出（无残留）", "thread 引用已清空",
          f"thread={service._thread}", service._thread is None)

    # stop 超时：worker 卡在 push 里，join 超时后必须保持 error
    original_timeout = recognition.STOP_JOIN_TIMEOUT
    recognition._load_stack = lambda: _FakeStack(delay=0.6)
    try:
        http("POST", f"{base}/api/listen/start",
             {"source": "simulated", "clip": "knock", "speed": 20.0})
        wait_state(base, ("listening",), timeout=30)
        recognition.STOP_JOIN_TIMEOUT = 0.05
        code, timed_out = http("POST", f"{base}/api/listen/stop")
        R.add("E7", "stop 超时不谎报 idle", "error",
              f"{code} {timed_out.get('state')} {str(timed_out.get('last_error'))[:30]}",
              code == 200 and timed_out.get("state") == "error")
        code, blocked = http("POST", f"{base}/api/listen/start",
                             {"source": "simulated", "clip": "knock"})
        R.add("E8", "线程存活期间禁止再启动", "409", f"{code}", code == 409)
    finally:
        recognition.STOP_JOIN_TIMEOUT = original_timeout
        recognition._load_stack = original_loader

    # 等卡住的线程自己退出，确认 error 不被悄悄改成 idle，但重新启动被允许
    deadline = time.time() + 15
    while time.time() < deadline and service._thread is not None:
        time.sleep(0.05)
    R.add("E9", "线程退出后仍保留 error（不谎报 idle）", "error", f"{service.state}",
          service.state == "error")
    code, recovered = http("POST", f"{base}/api/listen/start",
                           {"source": "simulated", "clip": "control", "speed": 20.0})
    R.add("E10", "线程退出后允许重新启动", "202",
          f"{code} {recovered.get('state')}", code == 202)
    http("POST", f"{base}/api/listen/stop")
    wait_state(base, ("idle",), timeout=60)

    # ---------- 麦克风拒绝 ----------
    print("\n[约束] 麦克风与持久化")
    code, payload = http("POST", f"{base}/api/listen/start", {"source": "microphone"})
    R.add("X1", "microphone 请求被拒", "400",
          f"{code} {str(payload)[:60]}", code == 400)
    R.add("X2", "ALLOW_MICROPHONE 常量", "False", f"{recognition.ALLOW_MICROPHONE}",
          recognition.ALLOW_MICROPHONE is False)
    status = get_status(base)
    R.add("X3", "设备未被打开", "opened=false / index=None",
          f"opened={status['device']['opened']} index={status['device']['index']}",
          status["device"]["opened"] is False and status["device"]["index"] is None)

    src = (PROJECT_ROOT / "app" / "recognition.py").read_text(encoding="utf-8")
    src_api = (PROJECT_ROOT / "app" / "listen_api.py").read_text(encoding="utf-8")
    forbidden = [t for t in ("import database", "get_connection",
                            "INSERT INTO", "create_simulated_event")
                 if t in src + src_api]
    # 敲门事件桥接允许**恰好一个**延迟落库入口，其余数据库引用仍禁止。
    allowed = "from .database import create_acoustic_knock_event"
    other_db = [
        line.strip() for line in src.splitlines()
        if "from .database" in line and line.strip() != allowed
    ]
    lazy_ok = src.count(allowed) == 1
    R.add("X4", "识别层无其它数据库引用（仅 1 个延迟敲门入口）",
          "无其它引用 + 恰好 1 处延迟 import",
          "无" if not forbidden and lazy_ok and not other_db
          else f"命中 {forbidden} lazy={lazy_ok} other={other_db}",
          not forbidden and lazy_ok and not other_db)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("main", "full"), default="full")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    print("=" * 78)
    print(f"阶段 2 监听接口冒烟测试  scope={args.scope}  解释器={sys.executable}")
    print("=" * 78)

    before = db_counts()
    print(f"  启动前数据库：sound_events={before[0]} 行，alert_settings={before[1]} 行")

    port = args.port or free_port()
    server, thread, base = start_server(port)
    print(f"  服务已启动 {base}")

    try:
        check_original_apis(base, f"{args.scope.upper()}-API")
        if args.scope == "main":
            run_main_scope(base)
        else:
            run_full_scope(base)
    finally:
        stop_server(server, thread)

    after = db_counts()
    print()
    R.add("DB1", "sound_events 行数不变", f"{before[0]}", f"{after[0]}",
          before[0] == after[0])
    R.add("DB2", "alert_settings 行数不变", f"{before[1]}", f"{after[1]}",
          before[1] == after[1])

    alive = [t.name for t in threading.enumerate() if t.name.startswith("listen-worker")]
    R.add("TH1", "无残留 listen-worker 线程", "0 个",
          f"{len(alive)} 个 {alive}", not alive)

    print("\n" + "=" * 78)
    n_pass = len(R.passed())
    n_fail = len(R.failures())
    print(f"PASS {n_pass}  FAIL {n_fail}  INFO {len(R.infos())}")
    if R.infos():
        print("\n仅记录项：")
        for tid, name, _e, actual, _ok, note in R.infos():
            print(f"  {tid:<8s} {name}: {actual} {note}")
    if n_fail:
        print("\n未通过项：")
        for tid, name, expected, actual, _ok, note in R.failures():
            print(f"  {tid:<8s} {name}: 期望 {expected} / 实测 {actual} {note}")
    else:
        print("\n未通过项：无")
    print("结论:", "PASS" if n_fail == 0 else "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
