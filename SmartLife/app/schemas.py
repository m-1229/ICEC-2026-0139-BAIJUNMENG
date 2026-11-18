"""API 请求和响应的数据结构。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


EventType = Literal["doorbell", "knock", "kettle", "alarm"]
EventSource = Literal["simulated", "acoustic"]


class SoundEvent(BaseModel):
    """统一的声音事件记录，区分模拟来源和声学识别来源。"""

    id: int
    event_type: EventType
    occurred_at: datetime
    confidence: float | None = Field(ge=0, le=1)
    source: EventSource


class SimulatedEventRequest(BaseModel):
    """创建模拟声音事件的请求。"""

    event_type: EventType
    confidence: float = Field(default=0.9, ge=0, le=1)


class AlertSettingUpdate(BaseModel):
    """更新一种声音类型的提醒开关。"""

    enabled: bool


class AlertSetting(BaseModel):
    """保存后的提醒开关设置。"""

    event_type: EventType
    enabled: bool
    updated_at: datetime


class TestAlertResponse(BaseModel):
    """手动测试提醒的响应；它不是声音识别结果，也不会被保存。"""

    kind: str
    message: str
    created_at: datetime
