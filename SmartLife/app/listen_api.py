"""监听控制接口（阶段 3：模拟素材 + 真实麦克风，仍不写数据库）。

四个端点
--------
GET  /api/listen/devices   只枚举输入设备，不打开任何音频端点
POST /api/listen/start     启动会话（202）：source=simulated | microphone
POST /api/listen/stop      停止并清理（幂等，永远 200）
GET  /api/listen/status    当前状态快照

错误语义（前端按这个实现即可）
------------------------------
400  参数/设备不支持：麦克风开关关闭、clip 未知、素材非 48 kHz、
     设备索引不在当前列表、设备被排除、设备做不到 48 kHz 单声道 float32
409  状态冲突：loading/listening/stopping 期间重复 start，或上一线程仍在退出
503  识别栈或音频后端不可用：缺 numpy / 缺模型 / 缺 sounddevice / 没有可用设备
"""

from fastapi import APIRouter, HTTPException, status as http_status

from .audio_devices import enumerate_input_devices
from .listen_schemas import DeviceListResponse, ListenStartRequest, ListenStatus
from .recognition import (
    ClipNotSupported,
    DeviceNotAllowed,
    ListenStateConflict,
    MicrophoneDisabled,
    MicrophoneUnavailable,
    RecognitionUnavailable,
    listen_service,
)

router = APIRouter(prefix="/api/listen", tags=["listen"])


@router.get("/devices", response_model=DeviceListResponse)
def list_input_devices() -> DeviceListResponse:
    """枚举输入设备。音频后端不可用时返回 available=false，不返回 5xx。"""
    return enumerate_input_devices()


@router.post(
    "/start",
    response_model=ListenStatus,
    status_code=http_status.HTTP_202_ACCEPTED,
)
def start_listening(request: ListenStartRequest) -> ListenStatus:
    """启动监听会话。模型加载与开流都在后台线程里进行。"""
    try:
        return listen_service.start(request)
    except (MicrophoneDisabled, DeviceNotAllowed, ClipNotSupported) as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except (RecognitionUnavailable, MicrophoneUnavailable) as exc:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ListenStateConflict as exc:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={"message": str(exc), "state": exc.state},
        ) from exc


@router.post("/stop", response_model=ListenStatus)
def stop_listening() -> ListenStatus:
    """停止监听并清理资源。重复调用、空闲调用都是幂等的 200。"""
    return listen_service.stop()


@router.get("/status", response_model=ListenStatus)
def listening_status() -> ListenStatus:
    """返回当前状态、设备、时长、帧数、触发数与内存事件。"""
    return listen_service.status()
