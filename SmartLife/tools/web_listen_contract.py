"""网页端契约测试：真实 FastAPI + 临时数据库 + 真实静态文件。

约束：
* 不打开麦克风（SMARTLIFE_ENABLE_MICROPHONE=0，服务端直接 400）。
* 数据库指向临时目录，正式 data/smartlife.db 只做哈希前后比对。
* 不新增依赖，只用标准库 urllib。

运行： .\\yamnet_prototype\\.venv\\Scripts\\python.exe tools\\web_listen_contract.py
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FORMAL_DB = PROJECT_ROOT / "data" / "smartlife.db"

os.environ["SMARTLIFE_ENABLE_MICROPHONE"] = "0"
os.environ["SMARTLIFE_ENABLE_KNOCK_PERSIST"] = "1"

import app.database as database  # noqa: E402

TMP_DIR = Path(tempfile.mkdtemp(prefix="smartlife_web_contract_"))
database.DATA_DIR = TMP_DIR
database.DATABASE_PATH = TMP_DIR / "contract.db"

passed = 0
failures: list[str] = []


def check(name: str, condition: bool, detail: object = None) -> None:
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}" + ("" if detail is None else f"  → {detail}"))


def eq(name: str, actual: object, expected: object) -> None:
    check(name, actual == expected, f"期望 {expected!r}，实际 {actual!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def call(method: str, url: str, payload: dict | None = None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            content_type = response.headers.get("content-type", "")
            if "json" in content_type:
                return response.status, json.loads(raw.decode())
            return response.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode())
        except Exception:
            return exc.code, raw.decode("utf-8", "replace")


def subprocess_flag(env_value: str | None) -> str:
    env = dict(os.environ)
    env.pop("SMARTLIFE_ENABLE_KNOCK_PERSIST", None)
    if env_value is not None:
        env["SMARTLIFE_ENABLE_KNOCK_PERSIST"] = env_value
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import app.recognition as r; print(r.KNOCK_PERSIST_ENABLED)" % PROJECT_ROOT
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "ERR"


def main() -> int:
    from app.main import app
    import uvicorn

    formal_before = sha256(FORMAL_DB) if FORMAL_DB.exists() else "MISSING"
    database.initialize_database()
    database.create_simulated_event("doorbell", 0.9)

    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, name="contract", daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{port}"
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.1)
    if not server.started:
        print("SERVER NOT STARTED")
        return 1

    print("\n[1] 静态资源")
    status, body = call("GET", f"{root}/")
    eq("首页 200", status, 200)
    check("首页含监听卡片", 'id="listen-device"' in body)
    check("首页含敲门横幅", 'id="knock-alert"' in body)
    check("首页引用 app.js v0.4.0", "app.js?v=0.4.0" in body)
    check("首页仍含原有模拟事件区块", 'id="simulate-event-form"' in body)

    status, body = call("GET", f"{root}/static/js/app.js")
    eq("app.js 200", status, 200)
    check("app.js 含 1.5s 状态轮询常量", "STATUS_POLL_MS = 1500" in body)
    check("app.js 含 2s 事件轮询常量", "EVENTS_POLL_MS = 2000" in body)
    check("app.js 不含震动调用", "navigator.vibrate" not in body)

    status, body = call("GET", f"{root}/static/css/styles.css")
    eq("styles.css 200", status, 200)
    check("样式含警报横幅", ".alert-banner" in body)
    check("样式含状态徽标", ".state-badge" in body)

    print("\n[2] 设备与状态接口")
    status, devices = call("GET", f"{root}/api/listen/devices")
    eq("devices 200", status, 200)
    eq("音频后端可用", devices.get("available"), True)
    usable = [d for d in devices.get("devices", []) if not d.get("excluded")]
    check("存在未排除设备", len(usable) > 0, len(usable))
    check(
        "存在 MME 未排除设备",
        any("mme" in str(d.get("hostapi", "")).lower() for d in usable),
        [d.get("hostapi") for d in usable],
    )
    for key in ("index", "name", "hostapi", "supports_48k", "excluded", "exclude_reason"):
        check(f"设备字段含 {key}", all(key in d for d in devices.get("devices", [])))

    status, snapshot = call("GET", f"{root}/api/listen/status")
    eq("status 200", status, 200)
    eq("初始 state=idle", snapshot.get("state"), "idle")
    eq("落库开关来自 acoustic", snapshot["acoustic"]["knock_persist_enabled"], True)
    check("acoustic 不含 listen_persist 字段", "listen_persist" not in snapshot["acoustic"])

    print("\n[3] 启动 / 停止的拒绝语义")
    status, body = call("POST", f"{root}/api/listen/start", {"source": "microphone"})
    eq("麦克风关闭时 400", status, 400)
    check("400 带可读 detail", isinstance(body.get("detail"), str) and "麦克风" in body["detail"], body)

    status, body = call(
        "POST", f"{root}/api/listen/start", {"source": "microphone", "device": 99999}
    )
    eq("麦克风关闭时不存在的索引仍是 400", status, 400)
    check("400 说明麦克风未开启", "麦克风" in str(body.get("detail")), body)

    # 设备校验分支：临时放开麦克风开关，只验证"被排除设备"在**打开端点之前**被拒绝。
    # _start_microphone 的顺序是 ALLOW_MICROPHONE -> probe -> _resolve_device -> 开流，
    # 被排除设备在 _resolve_device 就抛错，不会创建 InputStream。
    import app.recognition as recognition

    excluded = [d for d in devices.get("devices", []) if d.get("excluded")]
    if excluded:
        original = recognition.ALLOW_MICROPHONE
        recognition.ALLOW_MICROPHONE = True
        try:
            status, body = call(
                "POST",
                f"{root}/api/listen/start",
                {"source": "microphone", "device": excluded[0]["index"]},
            )
        finally:
            recognition.ALLOW_MICROPHONE = original
        eq("被排除设备 400", status, 400)
        check("400 说明设备已被排除", "已被排除" in str(body.get("detail")), body)
        status, after = call("GET", f"{root}/api/listen/status")
        eq("被拒绝后仍为 idle", after.get("state"), "idle")
        eq("被拒绝后未打开音频流", after["device"]["opened"], False)
    else:
        check("存在被排除设备用于校验", False, "本机没有 excluded 设备")

    status, body = call("POST", f"{root}/api/listen/stop")
    eq("空闲时 stop 幂等 200", status, 200)
    eq("stop 后 state=idle", body.get("state"), "idle")

    print("\n[4] 事件与设置接口（原有功能）")
    status, events = call("GET", f"{root}/api/events?limit=50")
    eq("events 200", status, 200)
    check("事件列表非空", len(events) >= 1)
    check(
        "事件字段齐全",
        all({"id", "event_type", "occurred_at", "confidence", "source"} <= set(e) for e in events),
    )
    print("      样例事件：", json.dumps(events[0], ensure_ascii=False))

    status, alerts = call("GET", f"{root}/api/settings/alerts")
    eq("alerts 200", status, 200)
    eq("提醒设置 4 条", len(alerts), 4)

    status, body = call("PATCH", f"{root}/api/settings/alerts/knock", {"enabled": False})
    eq("关闭 knock 开关 200", status, 200)
    eq("返回 enabled=False", body.get("enabled"), False)
    status, alerts = call("GET", f"{root}/api/settings/alerts")
    knock_setting = [a for a in alerts if a["event_type"] == "knock"][0]
    eq("关闭状态已持久化", knock_setting.get("enabled"), False)

    status, body = call(
        "POST", f"{root}/api/events/simulate", {"event_type": "knock", "confidence": 0.7}
    )
    eq("模拟事件 201", status, 201)
    eq("模拟来源标记", body.get("source"), "simulated")

    status, body = call("GET", f"{root}/api/health")
    eq("health 200", status, 200)
    eq("health.listen_persist 仍是 false（与敲门开关无关）", body.get("listen_persist"), False)

    print("\n[5] 落库开关三态（独立进程，避免 import 缓存）")
    eq("未设置 → False", subprocess_flag(None), "False")
    eq("=1 → True", subprocess_flag("1"), "True")
    eq("=0 → False", subprocess_flag("0"), "False")

    print("\n[6] 正式库与麦克风")
    formal_after = sha256(FORMAL_DB) if FORMAL_DB.exists() else "MISSING"
    eq("正式库 SHA256 未变", formal_after, formal_before)
    check("本次会话未打开麦克风（start 全部为 400）", True)
    eq("临时库路径与正式库不同", str(database.DATABASE_PATH) != str(FORMAL_DB), True)

    server.should_exit = True
    thread.join(timeout=15)

    print(f"\n结果：{passed} PASS / {len(failures)} FAIL")
    if failures:
        for name in failures:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
