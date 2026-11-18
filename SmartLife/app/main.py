"""SmartLife FastAPI 应用。"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .database import (
    create_simulated_event,
    get_alert_settings,
    get_sound_events,
    initialize_database,
    update_alert_setting,
)
from .listen_api import router as listen_router
from .recognition import (
    ALLOW_MICROPHONE,
    PERSIST_ENABLED,
    listen_service,
    microphone_capture_available,
)
from .schemas import (
    AlertSetting,
    AlertSettingUpdate,
    EventType,
    SimulatedEventRequest,
    SoundEvent,
    TestAlertResponse,
)


# 以当前 Python 文件的位置为基准，避免启动目录不同导致静态文件找不到。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时初始化数据库；关闭时先停掉监听线程，避免残留工作线程。"""
    initialize_database()
    try:
        yield
    finally:
        listen_service.shutdown()


app = FastAPI(
    title="SmartLife",
    description="面向听障人士的智能生活声音提醒辅助系统（基础开发阶段）",
    version="0.1.0",
    lifespan=lifespan,
)

# 网页所需的 CSS 和 JavaScript 通过 /static 地址提供。
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 监听控制接口（阶段 2：只支持模拟素材，不打开麦克风）。
app.include_router(listen_router)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """返回手机端首页。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health_check() -> dict[str, str | bool]:
    """供网页检查连接和新接口能力；当前不执行任何声音识别。"""
    return {
        "status": "ok",
        "message": "SmartLife 后端已连接。当前版本尚未实现声音识别。",
        "test_alert_available": True,
        "event_history_available": True,
        "alert_settings_available": True,
        "listen_available": listen_service.available(),
        "microphone_enabled": ALLOW_MICROPHONE,
        "microphone_capture_available": microphone_capture_available(),
        "listen_persist": PERSIST_ENABLED,
    }


@app.post("/api/test-alert", response_model=TestAlertResponse)
def create_test_alert() -> TestAlertResponse:
    """返回手动测试提醒，验证网页到后端的 POST 通信。"""
    return TestAlertResponse(
        kind="test",
        message="这是一条手动测试提醒，不是声音识别结果，也不会被保存。",
        created_at=datetime.now(timezone.utc),
    )


@app.post(
    "/api/events/simulate",
    response_model=SoundEvent,
    status_code=status.HTTP_201_CREATED,
)
def simulate_sound_event(request: SimulatedEventRequest) -> SoundEvent:
    """创建并保存一条显式标记为模拟来源的声音事件。"""
    return create_simulated_event(request.event_type, request.confidence)


@app.get("/api/events", response_model=list[SoundEvent])
def list_sound_events(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[SoundEvent]:
    """读取历史事件；分页参数有上限，避免一次返回过多记录。"""
    return get_sound_events(limit, offset)


@app.get("/api/settings/alerts", response_model=list[AlertSetting])
def list_alert_settings() -> list[AlertSetting]:
    """读取门铃、敲门、水壶和警报的提醒开关。"""
    return get_alert_settings()


@app.patch(
    "/api/settings/alerts/{event_type}",
    response_model=AlertSetting,
)
def set_alert_setting(
    event_type: EventType,
    request: AlertSettingUpdate,
) -> AlertSetting:
    """更新一种声音类型的提醒开关并持久化到 SQLite。"""
    return update_alert_setting(event_type, request.enabled)
