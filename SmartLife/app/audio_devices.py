"""输入设备枚举（只查询，不打开音频流）。

阶段 2 的约束
-------------
* 只用 `sounddevice.query_devices()` / `query_hostapis()`，**不调用** `InputStream`
  或 `check_input_settings()`，所以不会打开任何音频端点、不录音。
* `sounddevice` 或 PortAudio 不可用时返回 `available=False`，绝不抛 500：
  主项目在缺少识别依赖的环境里必须照常运行。

过滤与排序（对应"不得硬编码设备索引"的要求）
-------------------------------------------
1. 排除虚拟 / 回环设备（ToDesk Virtual Audio、VB-Audio、CABLE、Loopback、
   立体声混音、声音映射器、主声音捕获）；这些名字命中即标记 `excluded`。
2. 排除 WDM-KS 宿主：本机该宿主的设备名残缺（如"麦克风 ( )"），无法可靠识别。
3. 排序：采样率 48000 > 16000 > 44100 > 其它；宿主 WASAPI > DirectSound > MME；
   最后按名称稳定排序。
4. `target` 只是"阶段 3 建议使用的设备"，**本阶段不会被打开**。

注意：`default_samplerate` 是驱动声明的默认值，不等于"只支持该采样率"。
阶段 3 真正开流前必须用 `check_input_settings()` 复核（那时才允许触碰端点）。
"""

from __future__ import annotations

from .listen_schemas import DeviceInfo, DeviceListResponse

# 名称命中即排除（小写、去多余空格后比较）
EXCLUDED_NAME_PARTS: tuple[str, ...] = (
    "todesk",
    "virtual",
    "vb-audio",
    "vb cable",
    "cable output",
    "cable input",
    "voice meeter",
    "voicemeeter",
    "loopback",
    "stereo mix",
    "立体声混音",
    "声音映射器",
    "主声音捕获",
    "soundmapper",
    "primary sound capture",
)

# 整个宿主 API 排除：设备名残缺，无法判断是不是同一支物理麦克风
EXCLUDED_HOSTAPI_PARTS: tuple[str, ...] = ("wdm-ks",)

SAMPLERATE_PRIORITY: dict[int, int] = {48000: 0, 16000: 1, 44100: 2}
HOSTAPI_PRIORITY: tuple[str, ...] = ("wasapi", "directsound", "mme")


class DeviceEnumerationError(RuntimeError):
    """音频后端不可用或枚举过程失败。"""


def _import_backend():
    try:
        import sounddevice as sd  # 延迟导入：主 venv 没有这个包
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        raise DeviceEnumerationError(f"音频后端不可用（无法导入 sounddevice）：{exc}") from exc
    return sd


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def _exclude_reason(name: str, hostapi: str) -> str | None:
    """返回排除原因；None 表示该设备可以作为候选。"""
    norm_name = _norm(name)
    if not norm_name:
        return "设备名称为空"
    for part in EXCLUDED_NAME_PARTS:
        if _norm(part) in norm_name:
            return f"虚拟或回环设备（名称含「{part}」）"
    norm_host = _norm(hostapi)
    for part in EXCLUDED_HOSTAPI_PARTS:
        if part in norm_host:
            return f"宿主 API「{hostapi}」的设备名不可靠"
    return None


def _samplerate_rank(samplerate: float) -> int:
    try:
        rate = int(round(float(samplerate)))
    except (TypeError, ValueError):
        return 9
    return SAMPLERATE_PRIORITY.get(rate, 5 if rate >= 48000 else 8)


def _hostapi_rank(hostapi: str) -> int:
    norm = _norm(hostapi)
    for i, part in enumerate(HOSTAPI_PRIORITY):
        if part in norm:
            return i
    return len(HOSTAPI_PRIORITY)


def _sort_key(device: DeviceInfo) -> tuple:
    return (
        device.excluded,
        _samplerate_rank(device.default_samplerate),
        _hostapi_rank(device.hostapi),
        device.name.lower(),
        device.index,
    )


def enumerate_input_devices() -> DeviceListResponse:
    """枚举全部输入设备；失败时返回 available=False 而不抛异常。"""
    try:
        sd = _import_backend()
        raw_devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except DeviceEnumerationError as exc:
        return DeviceListResponse(available=False, detail=str(exc))
    except Exception as exc:  # PortAudioError 等
        return DeviceListResponse(
            available=False,
            detail=f"设备枚举失败：{type(exc).__name__}: {exc}",
        )

    devices: list[DeviceInfo] = []
    for index, raw in enumerate(raw_devices):
        try:
            channels = int(raw["max_input_channels"])
        except (KeyError, TypeError, ValueError):
            continue
        if channels <= 0:
            continue

        hostapi_index = int(raw.get("hostapi", -1))
        try:
            hostapi_name = str(hostapis[hostapi_index]["name"])
        except (IndexError, KeyError, TypeError):
            hostapi_name = f"hostapi-{hostapi_index}"

        name = str(raw.get("name", "")).strip()
        samplerate = raw.get("default_samplerate") or 0.0
        reason = _exclude_reason(name, hostapi_name)
        devices.append(
            DeviceInfo(
                index=index,
                name=name or "(未命名设备)",
                hostapi=hostapi_name,
                hostapi_index=hostapi_index,
                max_input_channels=channels,
                default_samplerate=float(samplerate),
                supports_48k=_samplerate_rank(samplerate) == 0,
                excluded=reason is not None,
                exclude_reason=reason,
            )
        )

    devices.sort(key=_sort_key)
    candidates = [d for d in devices if not d.excluded]
    target = candidates[0] if candidates else None

    detail = None
    if not devices:
        detail = "系统没有报告任何输入设备。"
    elif target is None:
        detail = "所有输入设备都被排除（虚拟设备或名称不可靠）。"

    return DeviceListResponse(
        available=True, devices=devices, target=target, detail=detail
    )
