#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""敲门正式事件桥接自测。

覆盖范围
--------
A 组：纯逻辑（脚本化检测器，不碰音频、不碰模型）——去重键、来源门禁、
      字段内容、写库失败隔离。
B 组：真实 `AcousticShadowDetector`（真实 DSP，但仍不开麦克风）——同一簇
      只写一条、新簇再写一条。
C 组：监听服务整链路（假输入流 + 假识别栈 + 真实检测器）——worker 线程里
      落库、写库失败不崩、停止后无残留线程。

安全约束
--------
* **只用临时数据库**：运行开始时把 `app.database.DATABASE_PATH` 指向
  系统临时文件，结束删除；正式库 `data/smartlife.db` 全程只读。
* 不打开真实麦克风（`_open_microphone` 被替换成假流）。
* 不录音、不保存 WAV、不上传、不改阈值与模型。

用法
----
    yamnet_prototype\\.venv\\Scripts\\python.exe -B tools\\knock_bridge_smoke.py

退出码 0 = 全部通过；1 = 有未通过项。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "tools"),
    str(PROJECT_ROOT / "yamnet_prototype"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# --------------------------------------------------------------------------- #
# 结果表
# --------------------------------------------------------------------------- #
class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, bool]] = []

    def add(self, tag: str, title: str, expect: str, actual,
            ok: bool, info: bool = False) -> None:
        self.rows.append((tag, title, expect, str(actual), bool(ok)))
        mark = "INFO" if info else ("PASS" if ok else "FAIL")
        print(f"  [{mark}] {tag:<4s} {title}")
        print(f"         期望: {expect}")
        print(f"         实际: {actual}")

    def summary(self) -> int:
        passed = sum(1 for r in self.rows if r[4])
        failed = len(self.rows) - passed
        print("\n" + "=" * 72)
        print(f"总计 {len(self.rows)} 项：PASS {passed} / FAIL {failed}")
        if failed:
            print("未通过：")
            for tag, title, expect, actual, ok in self.rows:
                if not ok:
                    print(f"  - {tag} {title} | 期望 {expect} | 实际 {actual}")
        print("=" * 72)
        return 0 if failed == 0 else 1


R = Results()


# --------------------------------------------------------------------------- #
# 临时数据库
# --------------------------------------------------------------------------- #
from app import database as DB  # noqa: E402
from app import recognition as REC  # noqa: E402
from app.listen_schemas import (  # noqa: E402
    AcousticCandidate as CandidateView,
    DeviceInfo,
    DeviceListResponse,
    ListenStartRequest,
)

PROD_DB = PROJECT_ROOT / "data" / "smartlife.db"


def prod_counts() -> tuple[int, int]:
    con = sqlite3.connect(f"file:{PROD_DB.as_posix()}?mode=ro", uri=True)
    try:
        return (
            int(con.execute("SELECT COUNT(*) FROM sound_events").fetchone()[0]),
            int(con.execute("SELECT COUNT(*) FROM alert_settings").fetchone()[0]),
        )
    finally:
        con.close()


def temp_counts(path: Path) -> int:
    con = sqlite3.connect(path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM sound_events").fetchone()[0])
    finally:
        con.close()


def install_temp_db(path: Path) -> None:
    """把数据层指向临时库；正式库只保留只读快照。"""
    DB.DATABASE_PATH = path
    DB.initialize_database()


def knock_view(cluster_id: str | None, impact_count: int = 2,
               start_s: float = 1.0) -> CandidateView:
    features: dict[str, float | int | str] = {
        "trigger_kind": "repeated_short_impacts",
        "impact_count": impact_count,
    }
    if cluster_id is not None:
        features["cluster_id"] = cluster_id
    return CandidateView(
        candidate_type="knock",
        start_time_s=start_s,
        time_s=round(start_s + 0.42, 3),
        duration_s=0.42,
        relative_db=18.5,
        reason="自测合成候选",
        features=features,
    )


def doorbell_view() -> CandidateView:
    return CandidateView(
        candidate_type="doorbell",
        start_time_s=1.0,
        time_s=1.3,
        duration_s=0.3,
        relative_db=16.0,
        reason="自测合成门铃候选",
        features={"trigger_kind": "stable_tonal_segment"},
    )


class ScriptedDetector:
    """按脚本吐候选的假检测器；`push` 一次消费一步。"""

    def __init__(self) -> None:
        self.steps: list[tuple[list[CandidateView], int]] = []
        self.reset_count = 0

    def step(self, candidates, reset_count: int | None = None) -> None:
        self.steps.append((list(candidates), self.reset_count if reset_count is None
                           else int(reset_count)))

    def push(self, _block):
        if not self.steps:
            return []
        candidates, generation = self.steps.pop(0)
        self.reset_count = generation
        return candidates

    def snapshot(self):
        return {"reset_count": self.reset_count, "frames_processed": 0}


def make_session(session_id: str, source: str = "microphone"):
    return REC._Session(
        session_id=session_id,
        source=source,
        clip=None if source == "microphone" else "knock",
        loop=False,
        speed=1.0,
        started_at=datetime.now(timezone.utc),
        started_mono=time.monotonic(),
    )


BLOCK = [0.0] * 960


def feed(svc, detector, session, candidates, reset_count: int | None = None) -> None:
    detector.step(candidates, reset_count)
    svc._shadow_push(detector, BLOCK, session)


# --------------------------------------------------------------------------- #
# A：纯逻辑
# --------------------------------------------------------------------------- #
def group_logic(db_path: Path) -> None:
    print("\n[A] 纯逻辑：去重键 / 来源门禁 / 字段 / 失败隔离")

    svc = REC.ListenService(knock_persist=True)
    detector = ScriptedDetector()

    session = make_session("sessA")

    # --- A1 同一簇首次出现：写一次 ---
    feed(svc, detector, session, [knock_view("knock-1", 2, 1.0)])
    rows = temp_counts(db_path)
    R.add("A1", "敲门簇首次出现写入 1 条正式事件",
          "rows=1 persisted=1", f"rows={rows} persisted={svc._knock_events_persisted}",
          rows == 1 and svc._knock_events_persisted == 1)

    # --- A2 同一簇后续更新：不重复写 ---
    feed(svc, detector, session, [knock_view("knock-1", 3, 1.22)])
    rows2 = temp_counts(db_path)
    views = [v for v in svc._acoustic_candidates if v.candidate_type == "knock"]
    R.add("A2", "同一簇后续更新不重复写库，只刷新内存视图",
          "rows=1 impact_count=3",
          f"rows={rows2} views={len(views)} "
          f"impact={[v.features.get('impact_count') for v in views]}",
          rows2 == 1 and len(views) == 1
          and int(views[0].features.get("impact_count", 0)) == 3)

    # --- A3 新簇：再写一次 ---
    feed(svc, detector, session, [knock_view("knock-2", 2, 5.0)])
    rows3 = temp_counts(db_path)
    R.add("A3", "新敲门簇再写一条", "rows=2",
          f"rows={rows3} persisted={svc._knock_events_persisted}",
          rows3 == 2 and svc._knock_events_persisted == 2)

    # --- A4 新会话：同编号簇仍可写 ---
    session_b = make_session("sessB")
    feed(svc, detector, session_b, [knock_view("knock-1", 2, 1.0)])
    rows4 = temp_counts(db_path)
    R.add("A4", "停止后新会话可正常创建新事件（knock-1 不冲突）",
          "rows=3", f"rows={rows4} persisted={svc._knock_events_persisted}",
          rows4 == 3 and svc._knock_events_persisted == 3)

    # --- A5 检测器 reset 后：新簇不被当成旧簇 ---
    feed(svc, detector, session_b, [knock_view("knock-1", 2, 1.0)], reset_count=1)
    rows5 = temp_counts(db_path)
    R.add("A5", "检测器 reset 后的新簇不被误判为旧簇",
          "rows=4", f"rows={rows5} persisted={svc._knock_events_persisted}",
          rows5 == 4 and svc._knock_events_persisted == 4)

    # reset 代数相同则仍然去重（防止把"代数"当成万能放行）
    feed(svc, detector, session_b, [knock_view("knock-1", 3, 1.2)], reset_count=1)
    rows5b = temp_counts(db_path)
    R.add("A5b", "同一 reset 代数下的同簇更新仍去重", "rows=4",
          f"rows={rows5b}", rows5b == 4)

    # --- A6 门铃候选：不写 ---
    feed(svc, detector, session_b, [doorbell_view()])
    rows6 = temp_counts(db_path)
    R.add("A6", "门铃候选不写入正式事件", "rows=4",
          f"rows={rows6}", rows6 == 4)

    # --- A7 缺 cluster_id 的 knock：不写 ---
    feed(svc, detector, session_b, [knock_view(None, 2, 9.0)])
    rows7 = temp_counts(db_path)
    R.add("A7", "缺少 cluster_id 的敲门候选不写（无法去重时宁可不写）",
          "rows=4", f"rows={rows7}", rows7 == 4)

    # --- A8 非 microphone 来源：不写 ---
    session_sim = make_session("sessSim", source="simulated")
    feed(svc, detector, session_sim, [knock_view("knock-9", 2, 2.0)])
    rows8 = temp_counts(db_path)
    R.add("A8", "模拟会话不写入正式事件", "rows=4",
          f"rows={rows8}", rows8 == 4)

    # --- A9 写库失败：不抛异常、错误计入；键不落地，下次更新可重试 ---
    saved = REC._persist_knock_event
    attempts = {"n": 0}

    def _flaky(occurred_at):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked (自测注入)")
        return saved(occurred_at)

    rows_before9 = temp_counts(db_path)
    persisted_before9 = svc._knock_events_persisted
    failed_key = (session_b.session_id, 0, "knock-3")
    REC._persist_knock_event = _flaky
    try:
        feed(svc, detector, session_b, [knock_view("knock-3", 2, 20.0)])
        snap = svc.status()
        errors_after_first = snap.acoustic.persist_errors
        last_err = snap.acoustic.persist_last_error
        rows_after_fail = temp_counts(db_path)
        persisted_fail = snap.acoustic.persisted_events
        key_done = failed_key in svc._knock_persisted
        inflight = list(svc._knock_inflight)

        # 同一簇后续更新 → 允许再次尝试，这次成功
        feed(svc, detector, session_b, [knock_view("knock-3", 3, 20.2)])
        snap2 = svc.status()
        rows_after_retry = temp_counts(db_path)
        persisted_retry = snap2.acoustic.persisted_events
        errors_after_retry = snap2.acoustic.persist_errors

        # 再更新一次 → 已写成功，不得重复写
        feed(svc, detector, session_b, [knock_view("knock-3", 4, 20.4)])
        rows_after_third = temp_counts(db_path)
    finally:
        REC._persist_knock_event = saved

    R.add("A9", "写库失败不抛异常，错误计入状态，去重键不落地",
          "errors=1 rows 不变 persisted 不变 键未完成 inflight 已清空",
          f"errors={errors_after_first} rows={rows_before9}->{rows_after_fail} "
          f"persisted={persisted_before9}->{persisted_fail} "
          f"key_done={key_done} inflight={inflight} last_error={last_err}",
          errors_after_first == 1 and rows_after_fail == rows_before9
          and persisted_fail == persisted_before9 and not key_done
          and not inflight and bool(last_err))
    R.add("A9b", "同一簇后续更新会重试，并在成功后写入 1 条",
          f"rows={rows_before9 + 1} persisted={persisted_before9 + 1} errors 仍为 1",
          f"attempts={attempts['n']} rows={rows_after_retry} "
          f"persisted={persisted_retry} errors={errors_after_retry}",
          rows_after_retry == rows_before9 + 1
          and persisted_retry == persisted_before9 + 1
          and errors_after_retry == 1)
    R.add("A9c", "重试成功后同一簇不再重复写入",
          "rows 与上一次相同",
          f"rows={rows_after_retry}->{rows_after_third} attempts={attempts['n']}",
          rows_after_third == rows_after_retry and attempts["n"] == 2)

    # --- A10 字段内容 ---
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT id, event_type, source, confidence, occurred_at "
            "FROM sound_events ORDER BY id LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    ok_field = (
        row is not None
        and row[1] == "knock"
        and row[2] == "acoustic"
        and row[3] is None
    )
    occurred = None
    if row is not None:
        try:
            occurred = datetime.fromisoformat(str(row[4]))
        except ValueError:
            occurred = None
    utc_ok = (
        occurred is not None
        and occurred.tzinfo is not None
        and occurred.utcoffset() == timezone.utc.utcoffset(None)
        and abs((datetime.now(timezone.utc) - occurred).total_seconds()) < 3600
    )
    R.add("A10", "事件字段：knock / acoustic / confidence=NULL / UTC 时间",
          "knock+acoustic+None+UTC",
          f"{row[1:4] if row else None} occurred_at={row[4] if row else None}",
          ok_field and utc_ok)

    # --- A11 /api/events 同源读取 ---
    events = DB.get_sound_events(20, 0)
    acoustic_knocks = [e for e in events if e.source == "acoustic"
                       and e.event_type == "knock"]
    R.add("A11", "数据层读取（/api/events 使用的同一函数）能取到新事件",
          ">=1 条 acoustic knock",
          f"{len(acoustic_knocks)} 条，首条 confidence={acoustic_knocks[0].confidence if acoustic_knocks else None}",
          bool(acoustic_knocks) and acoustic_knocks[0].confidence is None)

    # --- A12 模拟事件功能未受影响 ---
    before = temp_counts(db_path)
    sim = DB.create_simulated_event("knock", 0.91)
    after = temp_counts(db_path)
    R.add("A12", "模拟事件写入功能正常且来源仍为 simulated",
          "before+1 / source=simulated / confidence=0.91",
          f"{before}->{after} source={sim.source} conf={sim.confidence}",
          after == before + 1 and sim.source == "simulated"
          and abs(sim.confidence - 0.91) < 1e-9)

    # --- A13 关闭桥接开关后不写 ---
    svc_off = REC.ListenService(knock_persist=False)
    det_off = ScriptedDetector()
    before13 = temp_counts(db_path)
    feed(svc_off, det_off, make_session("sessOff"), [knock_view("knock-1", 2, 1.0)])
    after13 = temp_counts(db_path)
    R.add("A13", "knock_persist=False 时完全不写库", f"{before13} 不变",
          f"{before13}->{after13}", after13 == before13)


# --------------------------------------------------------------------------- #
# B：真实检测器（真实 DSP，不开麦克风）
# --------------------------------------------------------------------------- #
def synthetic_knock_block():
    import numpy as np

    n = 960
    t = np.arange(n, dtype=np.float32) / 48000.0
    env = np.exp(-t / 0.045).astype(np.float32)
    one = (
        (np.sin(2 * np.pi * 180 * t) + 0.45 * np.sin(2 * np.pi * 420 * t))
        * env * 0.20
    ).astype(np.float32)
    return one


def group_real_detector(db_path: Path) -> None:
    print("\n[B] 真实声学检测器（真实 DSP，不打开麦克风）")
    from acoustic_features import AcousticShadowDetector

    svc = REC.ListenService(knock_persist=True)
    session = make_session("sessReal")
    detector = AcousticShadowDetector(48000)
    knock = synthetic_knock_block()
    silence = [0.0] * 960
    before = temp_counts(db_path)

    def run_pattern(rounds: int, gap: int) -> None:
        for _ in range(rounds):
            svc._shadow_push(detector, knock, session)
            for _ in range(gap):
                svc._shadow_push(detector, silence, session)

    for _ in range(15):                      # 噪声底校准（≥10 帧）
        svc._shadow_push(detector, silence, session)
    run_pattern(3, 8)                        # 3 次冲击 → 同一个簇
    rows_b1 = temp_counts(db_path) - before
    cands = [c for c in svc.status().acoustic.candidates
             if c.candidate_type == "knock"]
    R.add("B1", "真实检测器产生的同一簇只写 1 条事件",
          "新增 1 行 impact_count>=2",
          f"新增={rows_b1} candidates={len(cands)} "
          f"impacts={[c.features.get('impact_count') for c in cands]}",
          rows_b1 == 1 and len(cands) == 1
          and int(cands[0].features.get("impact_count", 0)) >= 2)

    for _ in range(90):                      # >1.5 s 静音，结束当前簇
        svc._shadow_push(detector, silence, session)
    run_pattern(2, 8)                        # 新簇
    rows_b2 = temp_counts(db_path) - before
    ids = [c.features.get("cluster_id") for c in
           svc.status().acoustic.candidates if c.candidate_type == "knock"]
    R.add("B2", "真实检测器的第二个簇再写一条",
          "新增 2 行 且簇编号不同",
          f"新增={rows_b2} clusters={ids}",
          rows_b2 == 2 and len(set(ids)) == len(ids))


# --------------------------------------------------------------------------- #
# C：监听服务整链路（假输入流 + 假识别栈 + 真实检测器）
# --------------------------------------------------------------------------- #
class _Stats:
    def __init__(self) -> None:
        self.frames_produced = 0
        self.frames_scored = 0
        self.warmup_frames_dropped = 0
        self.stream_pos = 0
        self.buffered = 0


class _FakeRecognizer:
    def __init__(self) -> None:
        self.stats = _Stats()
        self.buffer = types.SimpleNamespace(reset=lambda: None)
        self.judge = types.SimpleNamespace(reset=lambda: None)
        self.resampler = None

    def reset_session(self) -> None:
        return None

    def push(self, block):
        self.stats.frames_produced += 1
        self.stats.stream_pos += len(block)
        return []

    def flush(self):
        return []


class _FakeStack:
    def config(self):
        return {"fake": 1}

    def new_recognizer(self, input_rate: int = 48000):
        return _FakeRecognizer()

    def new_acoustic(self, input_rate: int = 48000):
        from acoustic_features import AcousticShadowDetector
        return AcousticShadowDetector(input_rate)

    def load_clip(self, clip):
        raise AssertionError("本自测不需要模拟素材")


class _FakeStream:
    def __init__(self, feeder, *, realtime: float = 0.02, block: int = 960):
        self.feeder = feeder
        self.realtime = realtime
        self.block = block
        self.active = False
        self.closed = False
        self.calls = 0
        self.callback = None
        self.finished_callback = None
        self._thread = None

    def start(self) -> None:
        self.active = True
        self._thread = threading.Thread(target=self._run, name="fake-feeder",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        def sink_block(samples):
            if not self.active:
                return
            self.calls += 1
            self.callback(samples.reshape(-1, 1), int(samples.size), None, None)

        def sink_sleep(seconds):
            time.sleep(min(seconds, self.realtime))

        try:
            self.feeder(sink_block, sink_sleep)
        except Exception:
            pass

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False
        self.closed = True


class Patch:
    def __init__(self, **replacement):
        self.replacement = replacement
        self.saved = {}

    def __enter__(self):
        for name, value in self.replacement.items():
            self.saved[name] = getattr(REC, name)
            setattr(REC, name, value)
        return REC

    def __exit__(self, *_exc):
        for name, value in self.saved.items():
            setattr(REC, name, value)
        return False


def _fake_listing() -> DeviceListResponse:
    device = DeviceInfo(
        index=0, name="自测假麦克风", hostapi="MME", hostapi_index=0,
        max_input_channels=1, default_samplerate=48000.0,
        supports_48k=True, excluded=False, exclude_reason=None,
    )
    return DeviceListResponse(available=True, devices=[device], target=device)


def _knock_feeder(rounds: int, gap: int, lead: int):
    import numpy as np
    knock = synthetic_knock_block()
    silence = np.zeros(960, dtype=np.float32)

    def feeder(block, sleep):
        for _ in range(lead):
            block(silence)
            sleep(0.02)
        for _ in range(rounds):
            block(knock)
            sleep(0.02)
            for _ in range(gap):
                block(silence)
                sleep(0.02)
        for _ in range(30):
            block(silence)
            sleep(0.02)

    return feeder


def _start_fake_session(svc, feeder):
    opened: list = []

    def _open(device_index, sample_rate, blocksize, callback, finished_callback):
        stream = _FakeStream(feeder, realtime=0.02, block=blocksize)
        stream.callback = callback
        stream.finished_callback = finished_callback
        opened.append(stream)
        stream.start()          # 真实 _open_microphone 也在返回前 start()
        return stream

    patch = Patch(
        _open_microphone=_open,
        _check_input_settings=lambda *a, **k: None,
        probe_capability=lambda: (True, None),
        _load_stack=lambda: _FakeStack(),
        enumerate_input_devices=_fake_listing,
    )
    return patch, opened


def group_service(db_path: Path) -> None:
    print("\n[C] 监听服务整链路：假输入流 + 真实检测器 + 临时库")

    # --- C1 正常写入 ---
    svc = REC.ListenService(knock_persist=True)
    before = temp_counts(db_path)
    patch, opened = _start_fake_session(svc, _knock_feeder(3, 8, 15))
    with patch:
        svc.start(ListenStartRequest(source="microphone"))
        deadline = time.time() + 20
        status = svc.status()
        while time.time() < deadline:
            status = svc.status()
            if status.acoustic.persisted_events >= 1:
                break
            if status.state not in ("loading", "listening"):
                break
            time.sleep(0.05)
        state_during = status.state
        persisted = status.acoustic.persisted_events
        svc.stop()
    after = temp_counts(db_path)
    R.add("C1", "worker 线程内完成落库，状态机保持 listening",
          "state=listening 且 rows+1",
          f"state={state_during} persisted={persisted} rows {before}->{after}",
          state_during == "listening" and persisted == 1 and after == before + 1)
    R.add("C1b", "使用假输入流，未打开真实设备",
          "fake stream 已创建",
          f"streams={len(opened)}", bool(opened))
    R.add("C1c", "停止后无残留 listen-worker 线程",
          "无 listen-worker",
          str([t.name for t in threading.enumerate()
               if t.name.startswith("listen-worker")]),
          not any(t.name.startswith("listen-worker")
                  for t in threading.enumerate()))

    # --- C2 写库失败：worker 不崩 ---
    svc2 = REC.ListenService(knock_persist=True)
    before2 = temp_counts(db_path)

    def _boom(_occurred_at):
        raise sqlite3.OperationalError("disk I/O error (自测注入)")

    patch2, _ = _start_fake_session(svc2, _knock_feeder(3, 8, 15))
    saved = REC._persist_knock_event
    REC._persist_knock_event = _boom
    try:
        with patch2:
            svc2.start(ListenStartRequest(source="microphone"))
            deadline = time.time() + 20
            status2 = svc2.status()
            while time.time() < deadline:
                status2 = svc2.status()
                if status2.acoustic.persist_errors >= 1:
                    break
                if status2.state not in ("loading", "listening"):
                    break
                time.sleep(0.05)
            state2 = status2.state
            err = status2.acoustic.persist_errors
            msg = status2.acoustic.persist_last_error
            frame2 = status2.frames.produced
            svc2.stop()
    finally:
        REC._persist_knock_event = saved
    after2 = temp_counts(db_path)
    R.add("C2", "写库失败时 worker 不崩溃，错误可见",
          "state=listening errors>=1 last_error 非空",
          f"state={state2} errors={err} last_error={msg}",
          state2 == "listening" and err >= 1 and bool(msg))
    R.add("C2b", "写库失败时帧处理继续推进、库无新增",
          "produced>0 且 rows 不变",
          f"produced={frame2} rows {before2}->{after2}",
          frame2 > 0 and after2 == before2)
    R.add("C2c", "失败会话停止后无残留线程",
          "无 listen-worker",
          str([t.name for t in threading.enumerate()
               if t.name.startswith("listen-worker")]),
          not any(t.name.startswith("listen-worker")
                  for t in threading.enumerate()))


# --------------------------------------------------------------------------- #
# E：真实 HTTP 往返（/api/events 与 /api/listen/status）
# --------------------------------------------------------------------------- #
def group_http(db_path: Path) -> None:
    print("\n[E] 真实 HTTP 往返：/api/events 读取 + 状态字段")
    from listen_api_smoke import free_port, http, start_server, stop_server

    occurred = datetime.now(timezone.utc)
    written = DB.create_acoustic_knock_event(occurred)

    port = free_port()
    server, thread, base = start_server(port)
    try:
        code, payload = http("GET", f"{base}/api/events?limit=100")
        rows = payload if isinstance(payload, list) else []
        hit = [r for r in rows if int(r.get("id", -1)) == int(written.id)]
        R.add("E1", "GET /api/events 返回新写入的 acoustic 敲门事件",
              "200 且能按 id 取回，confidence=null",
              f"{code} 命中 {len(hit)} 条 "
              f"{hit[0] if hit else None}（该库共 {len(rows)} 条）",
              code == 200 and len(hit) == 1
              and hit[0].get("event_type") == "knock"
              and hit[0].get("source") == "acoustic"
              and hit[0].get("confidence") is None)

        code2, status = http("GET", f"{base}/api/listen/status")
        acoustic = status.get("acoustic", {}) if isinstance(status, dict) else {}
        R.add("E2", "监听状态暴露桥接开关与 YAMNet 落库开关的独立性",
              "knock_persist_enabled=false（默认关）, persist=false",
              f"{code2} {acoustic.get('knock_persist_enabled')} / "
              f"{status.get('persist') if isinstance(status, dict) else None}",
              code2 == 200
              and acoustic.get("knock_persist_enabled") is False
              and status.get("persist") is False)
        R.add("E3", "状态里桥接计数从 0 起算（未启动会话时不虚报）",
              "persisted_events=0",
              f"{acoustic.get('persisted_events')}",
              acoustic.get("persisted_events") == 0)
    finally:
        stop_server(server, thread)


# --------------------------------------------------------------------------- #
# F：事件时间（UTC 锚点 + 首击起点）
# --------------------------------------------------------------------------- #
ANCHOR = datetime(2026, 11, 18, 3, 30, 0, tzinfo=timezone.utc)


def group_time(db_path: Path) -> None:
    print("\n[F] 事件时间：UTC 锚点 + start_time_s，offset 不重复累加")
    saved = REC._persist_knock_event
    captured: list[datetime] = []

    def _capture(occurred_at):
        captured.append(occurred_at)
        return saved(occurred_at)

    # --- F1 锚点 + start_time_s ---
    svc = REC.ListenService(knock_persist=True)
    detector = ScriptedDetector()
    session = make_session("sessTime")
    session.listening_started_at = ANCHOR

    REC._persist_knock_event = _capture
    try:
        feed(svc, detector, session, [knock_view("knock-t1", 2, 3.75)])
    finally:
        REC._persist_knock_event = saved

    got = captured[-1] if captured else None
    delta = (got - ANCHOR).total_seconds() if got else None
    R.add("F1", "occurred_at = listening_started_at + start_time_s（首击起点）",
          "与锚点相差 3.750 s",
          f"{got.isoformat() if got else None} Δ={delta}",
          got is not None and delta is not None and abs(delta - 3.75) < 1e-9)

    # --- F2 时区 ---
    con = sqlite3.connect(db_path)
    try:
        stored = con.execute(
            "SELECT occurred_at FROM sound_events ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()
    parsed = datetime.fromisoformat(str(stored))
    R.add("F2", "落库时间是带 UTC 时区的 ISO 字符串",
          "+00:00 / tzinfo 非空 / 仍是锚点 + 3.75 s",
          f"{stored} tz={parsed.tzinfo} offset={parsed.utcoffset()}",
          parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
          and str(stored).endswith("+00:00")
          and abs((parsed - ANCHOR).total_seconds() - 3.75) < 1e-6)

    # --- F3 offset_s 只累加一次 ---
    # 检测器给的是"流开始以来"的原始秒，_shadow_push 会加一次 offset_s 形成
    # view.start_time_s；事件时间只能再用 view 的值，不能第二次加 offset_s。
    svc2 = REC.ListenService(knock_persist=True)
    detector2 = ScriptedDetector()
    session2 = make_session("sessOffset")
    session2.listening_started_at = ANCHOR
    with session2.mic.lock:
        session2.mic.offset_s = 6.0
    captured.clear()
    REC._persist_knock_event = _capture
    try:
        feed(svc2, detector2, session2, [knock_view("knock-o1", 2, 6.0)])
        matched = [
            v for v in svc2._acoustic_candidates
            if v.features.get("cluster_id") == "knock-o1"
        ]
    finally:
        REC._persist_knock_event = saved
    view = matched[0] if matched else None
    delta3 = (captured[-1] - ANCHOR).total_seconds() if captured else None
    R.add("F3", "offset_s 只累加一次：原始 6.0 s + offset 6.0 s → 事件时间 +12.0 s",
          "Δ=12.000（重复累加会得 18.000）",
          f"view.start_time_s={view.start_time_s if view else None} Δ={delta3}",
          view is not None and delta3 is not None
          and abs(delta3 - 12.0) < 1e-6
          and abs(float(view.start_time_s) - 12.0) < 1e-6)


# --------------------------------------------------------------------------- #
# G：环境变量（独立进程，验证导入时读取的真实行为）
# --------------------------------------------------------------------------- #
ENV_PROBE_TEMPLATE = r'''
import json, os, sqlite3, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, r"__ROOT__")
from app import database as DB
from app import recognition as REC
from app.listen_schemas import AcousticCandidate

handle = tempfile.NamedTemporaryFile(
    prefix="smartlife_envprobe_", suffix=".db", delete=False
)
handle.close()
DB.DATABASE_PATH = Path(handle.name)
DB.initialize_database()

svc = REC.listen_service
session = REC._Session(
    session_id="envprobe", source="microphone", clip=None,
    loop=False, speed=1.0,
    started_at=datetime.now(timezone.utc), started_mono=0.0,
)
session.listening_started_at = datetime.now(timezone.utc)
view = AcousticCandidate(
    candidate_type="knock", start_time_s=1.0, time_s=1.4,
    duration_s=0.4, relative_db=20.0, reason="env probe",
    features={"cluster_id": "knock-1", "impact_count": 2},
)


class _Det:
    reset_count = 0

    def push(self, _block):
        return [view]

    def snapshot(self):
        return {"reset_count": 0}


svc._shadow_push(_Det(), [0.0] * 960, session)
con = sqlite3.connect(handle.name)
rows = int(con.execute("SELECT COUNT(*) FROM sound_events").fetchone()[0])
con.close()
os.unlink(handle.name)
print(json.dumps({
    "const": bool(REC.KNOCK_PERSIST_ENABLED),
    "singleton": bool(svc._knock_persist),
    "rows": rows,
}))
'''

ENV_PROBE = ENV_PROBE_TEMPLATE.replace("__ROOT__", str(PROJECT_ROOT))


def _run_env_probe(value: str | None) -> dict:
    env = dict(os.environ)
    env.pop("SMARTLIFE_ENABLE_KNOCK_PERSIST", None)
    if value is not None:
        env["SMARTLIFE_ENABLE_KNOCK_PERSIST"] = value
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-B", "-c", ENV_PROBE],
        cwd=str(PROJECT_ROOT), env=env,
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "")[-500:]}
    lines = [ln for ln in (proc.stdout or "").splitlines()
             if ln.strip().startswith("{")]
    if not lines:
        return {"error": "子进程未输出 JSON"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {"error": f"{exc}: {lines[-1][:200]}"}


def group_env(db_path: Path) -> None:
    print("\n[G] 环境变量：独立进程验证 未设置 / 0 / 1（不是构造参数）")
    cases = [
        ("G1", None, False, "未设置"),
        ("G2", "0", False, "=0"),
        ("G3", "1", True, "=1"),
    ]
    for tag, value, want, label in cases:
        out = _run_env_probe(value)
        R.add(tag, f"SMARTLIFE_ENABLE_KNOCK_PERSIST{label}：常量/单例/是否落库",
              f"const={want} singleton={want} rows={1 if want else 0}",
              out,
              out.get("const") is want and out.get("singleton") is want
              and out.get("rows") == (1 if want else 0))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="敲门正式事件桥接自测")
    parser.add_argument("--keep-db", action="store_true",
                        help="保留临时数据库文件（默认删除）")
    args = parser.parse_args()

    prod_before = prod_counts()
    print(f"正式数据库只读快照：sound_events={prod_before[0]} "
          f"alert_settings={prod_before[1]}")

    handle = tempfile.NamedTemporaryFile(
        prefix="smartlife_knock_bridge_", suffix=".db", delete=False
    )
    handle.close()
    db_path = Path(handle.name)
    print(f"临时数据库：{db_path}")
    install_temp_db(db_path)

    try:
        group_logic(db_path)
        group_time(db_path)
        group_real_detector(db_path)
        group_service(db_path)
        group_env(db_path)
        group_http(db_path)
    finally:
        prod_after = prod_counts()
        R.add("D1", "正式数据库未被写入",
              f"{prod_before} 不变", f"{prod_after}",
              prod_after == prod_before)
        if not args.keep_db:
            try:
                db_path.unlink()
            except OSError:
                pass

    return R.summary()


if __name__ == "__main__":
    sys.exit(main())
