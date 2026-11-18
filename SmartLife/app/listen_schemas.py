"""监听接口的数据结构（阶段 2）。

与 `app/schemas.py` 分开存放，原因：
1. 阶段 2 不修改 `schemas.py`，避免碰 `EventSource`（数据库 CHECK 约束配套）。
2. 监听相关结构只在本阶段新增，集中一处便于阶段 4 迁移时整体调整。

约定
----
* `ListenEvent` **没有 id 字段**：阶段 2 事件只存在于内存，没有数据库主键。
* `persist` 恒为 False，由服务端固定，客户端无法通过请求参数改变。
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# 状态机：idle -> loading -> listening -> stopping -> idle，任一步异常落 error
ListenState = Literal["idle", "loading", "listening", "stopping", "error"]

# 阶段 2 只允许模拟输入；microphone 会被服务端拒绝
ListenSource = Literal["simulated", "microphone"]

# 服务端固定映射的三个素材，客户端只能传这三个名字
SimulatedClip = Literal["doorbell", "knock", "control"]


class DeviceInfo(BaseModel):
    """一个输入设备候选（只来自枚举，不代表已经打开）。"""

    index: int
    name: str
    hostapi: str
    hostapi_index: int
    max_input_channels: int
    default_samplerate: float
    supports_48k: bool
    excluded: bool
    exclude_reason: str | None = None


class DeviceListResponse(BaseModel):
    """设备枚举结果。`available=False` 表示音频后端不可用，不是请求失败。"""

    available: bool
    devices: list[DeviceInfo] = Field(default_factory=list)
    target: DeviceInfo | None = None
    detail: str | None = None
    note: str = "本接口只查询设备列表，不打开麦克风、不录音、不启动监听。"


class ListenStartRequest(BaseModel):
    """启动监听会话的请求。

    * `source="simulated"`：喂固定素材，用于回归与联调演练。
    * `source="microphone"`：打开真实输入流（阶段 3）。`device` 是
      `GET /api/listen/devices` 返回列表里的 `index`；服务端会重新枚举并校验，
      **不信任客户端传来的任意索引**。留空则使用筛选后的推荐设备。
    """

    source: ListenSource = "simulated"
    clip: SimulatedClip = "doorbell"
    loop: bool = False
    speed: float = Field(
        default=1.0,
        ge=0.1,
        le=20.0,
        description="模拟喂入速度倍数，只改变节拍，不改变样本内容。",
    )
    device: int | None = Field(
        default=None,
        ge=0,
        description="麦克风设备索引（仅 source=microphone 时生效）；"
        "必须在当前设备列表中且未被排除。",
    )


class ListenEvent(BaseModel):
    """一次触发的内存记录。seq 在会话内从 1 递增，新会话重新计数。"""

    seq: int
    event_type: str
    score: float
    occurred_at: datetime
    time_s: float
    frame_index: int


class FrameStats(BaseModel):
    """窗口与判定计数，来自 StreamingRecognizer.stats。"""

    produced: int = 0
    scored: int = 0
    warmup_dropped: int = 0


class SessionDevice(BaseModel):
    """本会话使用的输入描述。

    * 模拟会话：`index=None`、`hostapi="none"`、`opened=False`（没有真实端点）。
    * 麦克风会话：`index/name/hostapi/sample_rate` 全部来自真实设备，
      只有输入流真正 start() 成功后 `opened` 才为 True。
    """

    index: int | None = None
    name: str
    hostapi: str
    sample_rate: int
    opened: bool = False
    channels: int | None = None
    dtype: str | None = None


class MicStats(BaseModel):
    """真实麦克风采集的队列、异常计数与电平（模拟会话全为 0）。

    dropped_blocks / overflow_events 用来判断"有没有丢过音频"：
    只要发生过溢出，服务端就会清空窗口缓冲与 3 帧判定缓存并累加
    `timeline_offset_s`，因此 `time_s` 仍然是"监听开始以来的秒数"，
    不会因为丢块而倒退或错位。

    电平是**诊断字段**，全部来自音频回调里对真实样本的实测，
    不参与任何判定：
    * `level_peak` / `level_peak_dbfs`：最近一个采集块的峰值（线性 / dBFS）。
    * `level_peak_recent`：最近 1 秒（50 块 × 20 ms）内的最大峰值。
    * `level_peak_max`：本次会话的最大峰值。
    * `level_rms_dbfs`：最近 1 秒窗口的 RMS（dBFS）。
    * `zero_blocks`：**数字零**块数（块内样本全为 0.0）——端点不再供数时会出现。
    * `silent_blocks`：非零但峰值 < 1e-6（低于 -120 dBFS）的"真实静音"块数。
    * `level_blocks`：已统计电平的块数。

    判读：`level_peak≈1e-5..1e-3` 且 `zero_blocks=0` 是**正常底噪**；
    `zero_blocks` 快速增长说明**端点异常**；`level_peak_recent` 出现尖峰才是**真有声音**。
    """

    block_samples: int = 0
    queue_max_blocks: int = 0
    queue_depth: int = 0
    dropped_blocks: int = 0
    overflow_events: int = 0
    resync_count: int = 0
    callback_errors: int = 0
    input_flags: int = 0
    timeline_offset_s: float = 0.0
    # ---- 电平诊断（实测值，不参与判定） ----
    level_peak: float = 0.0
    level_peak_dbfs: float = -160.0
    level_peak_recent: float = 0.0
    level_peak_max: float = 0.0
    level_rms_dbfs: float = -160.0
    level_blocks: int = 0
    zero_blocks: int = 0
    silent_blocks: int = 0


class FrameScore(BaseModel):
    """单帧的模型分数（诊断字段，不参与判定）。

    * `smoothed` 是 3 帧滑动均值（`smooth_n=SMOOTH_N`），也就是触发判据用的
      那个数；窗口不足 3 帧时为 None。
    * `raw` 是该帧的原始分组分数。
    * `triggered` 只表示这一帧是否产生了触发（与原逻辑完全一致）。
    * `time_s` 与事件同口径：相对监听开始，且已加丢块偏移。
    """

    frame_index: int
    window_index: int
    time_s: float
    doorbell: float | None = None
    knock: float | None = None
    raw_doorbell: float | None = None
    raw_knock: float | None = None
    triggered: bool = False


class AcousticCandidate(BaseModel):
    """声学特征影子候选。

    这不是正式提醒，也不是 YAMNet 置信度。`features` 保存形成候选时的关键
    规则值，用于真实测试后判断误报和漏报。
    """

    candidate_type: Literal["doorbell", "knock"]
    start_time_s: float
    time_s: float
    duration_s: float
    relative_db: float
    reason: str
    features: dict[str, float | int | str] = Field(default_factory=dict)


class AcousticStats(BaseModel):
    """声学影子检测状态；始终不参与 YAMNet 的正式触发。

    本阶段的唯一变化：`knock` 候选在被确认后可以写入正式事件表。
    因此这里增加一组**桥接计数**，用于证明"同一簇只写一次"以及
    "数据库失败不影响监听"：

    * `knock_persist_enabled`：服务端是否启用了敲门事件桥接（固定由服务端决定）。
    * `persisted_events`：本次会话成功写入的敲门事件条数。
    * `persist_errors` / `persist_last_error`：写入失败次数与最近一次错误。
    * `last_event_id`：最近写入的 `sound_events.id`，便于和 `/api/events` 对账。

    门铃候选、YAMNet 触发、模拟会话都不会改变这些计数。
    """

    shadow_only: bool = True
    frames_processed: int = 0
    samples_processed: int = 0
    noise_floor_dbfs: float = -160.0
    calibrated: bool = False
    current_relative_db: float | None = None
    current_spectral_flatness: float | None = None
    current_spectral_centroid_hz: float | None = None
    current_spectral_similarity: float | None = None
    current_tonal: bool = False
    pending_impacts: int = 0
    recent_impacts: int = 0
    last_reset_reason: str = "session"
    reset_count: int = 0
    session_resets: int = 0
    resync_resets: int = 0
    errors: int = 0
    last_error: str | None = None
    # ---- 敲门正式事件桥接（本阶段新增） ----
    knock_persist_enabled: bool = False
    persisted_events: int = 0
    persist_errors: int = 0
    persist_last_error: str | None = None
    last_event_id: int | None = None
    candidates: list[AcousticCandidate] = Field(default_factory=list)


class ListenStatus(BaseModel):
    """GET /api/listen/status 的完整快照，也是 start/stop 的响应。"""

    state: ListenState
    source: str | None = None
    clip: str | None = None
    persist: bool = False
    microphone_enabled: bool = False
    session_id: str | None = None
    started_at: datetime | None = None
    uptime_s: float = 0.0
    listening_s: float = 0.0
    device: SessionDevice | None = None
    frames: FrameStats = Field(default_factory=FrameStats)
    mic: MicStats = Field(default_factory=MicStats)
    #: 最近若干帧的逐帧分数（诊断用，环形缓冲；即使没有触发也会出现）。
    frame_scores: list[FrameScore] = Field(default_factory=list)
    #: 本会话各组 3 帧均值的峰值（诊断用）；没有判定记录时为空。
    peak_smoothed: dict[str, float] = Field(default_factory=dict)
    #: 声学特征影子候选与实时指标；绝不参与正式提醒触发。
    acoustic: AcousticStats = Field(default_factory=AcousticStats)
    stream_pos: int = 0
    buffered: int = 0
    triggers: int = 0
    events: list[ListenEvent] = Field(default_factory=list)
    last_error: str | None = None
    stop_reason: str | None = None
    config: dict[str, float | int | str] = Field(default_factory=dict)
