// 页面仅展示后端保存的数据；模拟事件不代表真实声音识别。
const EVENT_LABELS = {
  doorbell: "门铃",
  knock: "敲门",
  kettle: "水壶",
  alarm: "警报",
};

const statusElement = document.querySelector("#connection-status");
const refreshButton = document.querySelector("#refresh-button");
const testAlertCard = document.querySelector("#test-alert-card");
const testAlertButton = document.querySelector("#test-alert-button");
const testAlertStatus = document.querySelector("#test-alert-status");
const simulateEventForm = document.querySelector("#simulate-event-form");
const simulateEventButton = document.querySelector("#simulate-event-button");
const simulateEventStatus = document.querySelector("#simulate-event-status");
const settingsList = document.querySelector("#settings-list");
const settingsStatus = document.querySelector("#settings-status");
const eventList = document.querySelector("#event-list");
const eventsStatus = document.querySelector("#events-status");
const refreshEventsButton = document.querySelector("#refresh-events-button");
const knockAlertHint = document.querySelector("#knock-alert-hint");

// ---- 监听控制相关元素 ----
const listenDeviceSelect = document.querySelector("#listen-device");
const listenStartButton = document.querySelector("#listen-start-button");
const listenStopButton = document.querySelector("#listen-stop-button");
const refreshDevicesButton = document.querySelector("#refresh-devices-button");
const listenStateBadge = document.querySelector("#listen-state");
const listenMeta = document.querySelector("#listen-meta");
const listenPersistStatus = document.querySelector("#listen-persist-status");
const listenStatus = document.querySelector("#listen-status");
const knockAlert = document.querySelector("#knock-alert");
const knockAlertTime = document.querySelector("#knock-alert-time");
const knockAlertDismiss = document.querySelector("#knock-alert-dismiss");

// 状态文案：与后端 ListenState 一一对应（idle/loading/listening/stopping/error）。
const LISTEN_STATE_TEXT = {
  idle: "未监听",
  loading: "启动中",
  listening: "正在监听",
  stopping: "停止中",
  error: "异常",
};

// 监听期间的状态轮询间隔与事件轮询间隔（毫秒）。
const STATUS_POLL_MS = 1500;
const EVENTS_POLL_MS = 2000;

const listenUi = {
  state: "idle",
  sessionId: null,
  lastError: null,
  // 设备
  devicesLoaded: false,
  deviceUsable: false,
  // 落库开关（只来自 status.acoustic.knock_persist_enabled）
  persistEnabled: null,
  persistedEvents: 0,
  persistErrors: 0,
  // 定时器
  statusTimer: null,
  eventsTimer: null,
  // 事件基线：首次加载时记录最大 id，之后只对更大的 id 提醒
  lastEventId: 0,
  baselineReady: false,
  alertedIds: new Set(),
  // 敲门提醒开关（来自 /api/settings/alerts）
  knockEnabled: true,
  knockEnabledKnown: false,
};

function eventLabel(eventType) {
  return EVENT_LABELS[eventType] || eventType;
}

function formatEventTime(value) {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return String(value);
  }
  return timestamp.toLocaleString();
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const rest = String(total % 60).padStart(2, "0");
  return `${minutes}:${rest}`;
}

// 后端错误体有两种形状：{detail: "文本"} 与 {detail: {message, state}}。
function extractErrorDetail(payload, fallbackStatus) {
  if (payload && typeof payload === "object") {
    const detail = payload.detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
    if (detail && typeof detail === "object" && detail.message) {
      return detail.state ? `${detail.message}（当前状态：${detail.state}）` : detail.message;
    }
    if (payload.message) {
      return String(payload.message);
    }
  }
  return `服务器返回错误状态：${fallbackStatus}`;
}

async function checkBackend() {
  statusElement.textContent = "正在连接后端…";
  statusElement.dataset.state = "loading";
  refreshButton.disabled = true;

  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`服务器返回错误状态：${response.status}`);
    }

    const result = await response.json();
    statusElement.textContent = result.message;
    statusElement.dataset.state = "success";
    testAlertCard.hidden = result.test_alert_available !== true;

    await Promise.all([loadAlertSettings(), loadEvents()]);
  } catch (error) {
    statusElement.textContent = `连接失败：${error.message} 请确认后端服务正在运行。`;
    statusElement.dataset.state = "error";
    testAlertCard.hidden = true;
  } finally {
    refreshButton.disabled = false;
  }
}

async function sendTestAlert() {
  testAlertStatus.textContent = "正在发送测试提醒…";
  testAlertStatus.dataset.state = "loading";
  testAlertButton.disabled = true;

  try {
    const response = await fetch("/api/test-alert", { method: "POST" });
    if (!response.ok) {
      throw new Error(`服务器返回错误状态：${response.status}`);
    }

    const result = await response.json();
    const time = new Date(result.created_at).toLocaleString();
    testAlertStatus.textContent = `${result.message}（${time}）`;
    testAlertStatus.dataset.state = "success";
  } catch (error) {
    testAlertStatus.textContent = `测试提醒发送失败：${error.message}`;
    testAlertStatus.dataset.state = "error";
  } finally {
    testAlertButton.disabled = false;
  }
}

async function loadAlertSettings() {
  settingsStatus.textContent = "正在读取提醒设置…";
  settingsStatus.dataset.state = "loading";

  try {
    const response = await fetch("/api/settings/alerts");
    if (!response.ok) {
      throw new Error(`服务器返回错误状态：${response.status}`);
    }

    const settings = await response.json();
    settingsList.replaceChildren();

    for (const setting of settings) {
      const row = document.createElement("label");
      row.className = "setting-row";

      const name = document.createElement("span");
      name.textContent = `${eventLabel(setting.event_type)}提醒`;

      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.checked = setting.enabled;
      toggle.setAttribute("aria-label", `${eventLabel(setting.event_type)}提醒开关`);
      toggle.addEventListener("change", () => {
        updateAlertSetting(setting.event_type, toggle);
      });

      row.append(name, toggle);
      settingsList.append(row);

      if (setting.event_type === "knock") {
        listenUi.knockEnabled = setting.enabled === true;
        listenUi.knockEnabledKnown = true;
        renderKnockAlertHint();
      }
    }

    settingsStatus.textContent = "设置已从 SQLite 加载。";
    settingsStatus.dataset.state = "success";
  } catch (error) {
    settingsStatus.textContent = `读取提醒设置失败：${error.message}`;
    settingsStatus.dataset.state = "error";
  }
}

async function updateAlertSetting(eventType, toggle) {
  const previousValue = !toggle.checked;
  toggle.disabled = true;
  settingsStatus.textContent = `正在保存${eventLabel(eventType)}提醒设置…`;
  settingsStatus.dataset.state = "loading";

  try {
    const response = await fetch(
      `/api/settings/alerts/${encodeURIComponent(eventType)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: toggle.checked }),
      },
    );
    if (!response.ok) {
      throw new Error(`服务器返回错误状态：${response.status}`);
    }

    const savedSetting = await response.json();
    toggle.checked = savedSetting.enabled;
    settingsStatus.textContent =
      `${eventLabel(eventType)}提醒已${savedSetting.enabled ? "开启" : "关闭"}并保存。`;
    settingsStatus.dataset.state = "success";

    if (eventType === "knock") {
      listenUi.knockEnabled = savedSetting.enabled === true;
      listenUi.knockEnabledKnown = true;
      renderKnockAlertHint();
      // 关闭提醒开关后，已经弹出的横幅不应继续占屏。
      if (!savedSetting.enabled) {
        hideKnockAlert();
      }
    }
  } catch (error) {
    toggle.checked = previousValue;
    settingsStatus.textContent = `保存提醒设置失败：${error.message}`;
    settingsStatus.dataset.state = "error";
  } finally {
    toggle.disabled = false;
  }
}

function renderEventList(events) {
  eventList.replaceChildren();

  for (const event of events) {
    const item = document.createElement("li");
    item.className = "event-item";

    const timestamp = new Date(event.occurred_at);
    const timeText = Number.isNaN(timestamp.getTime())
      ? event.occurred_at
      : timestamp.toLocaleString();
    const confidenceText = event.confidence === null
      ? "置信度未提供"
      : `置信度 ${Math.round(event.confidence * 100)}%`;
    const sourceText = event.source === "simulated" ? "模拟事件" : event.source;

    item.textContent =
      `#${event.id} · ${eventLabel(event.event_type)} · ${timeText} · ${confidenceText} · ${sourceText}`;
    eventList.append(item);
  }
}

// 只有页面首次成功读取历史时才建立基线：之后不再回退或覆盖，
// 否则会把"已经到达但还没提醒"的新事件当成旧记录吞掉。
function captureEventBaseline(events) {
  if (listenUi.baselineReady || !Array.isArray(events)) {
    return;
  }
  listenUi.lastEventId = events.reduce(
    (max, event) => Math.max(max, Number(event.id) || 0),
    0,
  );
  listenUi.baselineReady = true;
}

// 把已读取到的事件标记为"已见过"，避免下次轮询把它们当成新事件补弹。
function markEventsSeen(events) {
  if (!Array.isArray(events)) {
    return;
  }
  const maxId = events.reduce(
    (max, event) => Math.max(max, Number(event.id) || 0),
    0,
  );
  if (maxId > listenUi.lastEventId) {
    listenUi.lastEventId = maxId;
  }
}

async function loadEvents() {
  eventsStatus.textContent = "正在读取历史记录…";
  eventsStatus.dataset.state = "loading";

  try {
    const response = await fetch("/api/events?limit=50");
    if (!response.ok) {
      throw new Error(`服务器返回错误状态：${response.status}`);
    }

    const events = await response.json();
    renderEventList(events);
    captureEventBaseline(events);
    // 只在没有活动会话时推进"已见"水位：监听中手动刷新不能吞掉待提醒的新事件。
    if (listenUi.state === "idle" || listenUi.state === "error") {
      markEventsSeen(events);
    }

    if (events.length === 0) {
      eventsStatus.textContent = "暂无事件记录。可以创建一条模拟事件进行测试。";
      eventsStatus.dataset.state = "success";
      return;
    }

    eventsStatus.textContent = `已加载 ${events.length} 条记录。`;
    eventsStatus.dataset.state = "success";
  } catch (error) {
    eventsStatus.textContent = `读取历史记录失败：${error.message}`;
    eventsStatus.dataset.state = "error";
  }
}

async function createSimulatedEvent(event) {
  event.preventDefault();
  simulateEventButton.disabled = true;
  simulateEventStatus.textContent = "正在保存模拟事件…";
  simulateEventStatus.dataset.state = "loading";

  const formData = new FormData(simulateEventForm);
  const eventType = formData.get("event_type");
  const confidence = Number(formData.get("confidence"));

  try {
    const response = await fetch("/api/events/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_type: eventType, confidence }),
    });
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || `服务器返回错误状态：${response.status}`);
    }

    const createdEvent = await response.json();
    simulateEventStatus.textContent =
      `${eventLabel(createdEvent.event_type)}模拟事件已写入数据库（来源：模拟）。`;
    simulateEventStatus.dataset.state = "success";
    await loadEvents();
  } catch (error) {
    simulateEventStatus.textContent = `创建模拟事件失败：${error.message}`;
    simulateEventStatus.dataset.state = "error";
  } finally {
    simulateEventButton.disabled = false;
  }
}

// --------------------------------------------------------------------------
// 麦克风设备
// --------------------------------------------------------------------------
function isMmeDevice(device) {
  return String(device.hostapi || "").toLowerCase().includes("mme");
}

// 默认优先级：
// 1) MME 且声明支持 48 kHz（本机通常没有）
// 2) MME（驱动常把默认采样率报成 44.1 kHz，但服务端固定以 48 kHz 打开并复核，
//    详见 app/audio_devices.py 里 default_samplerate 的说明，故排在 3) 之前）
// 3) 其它声明支持 48 kHz 的设备
// 4) 其余未被排除的设备
function deviceRank(device) {
  const mme = isMmeDevice(device);
  if (mme && device.supports_48k) {
    return 0;
  }
  if (mme) {
    return 1;
  }
  if (device.supports_48k) {
    return 2;
  }
  return 3;
}

function describeDevice(device) {
  const parts = [device.hostapi];
  if (device.supports_48k) {
    parts.push("48 kHz");
  }
  parts.push(`#${device.index}`);
  return `${device.name}（${parts.join(" · ")}）`;
}

function setDeviceUsable(usable) {
  listenUi.deviceUsable = usable;
  renderListenButtons();
}

async function loadDevices() {
  listenStatus.textContent = "正在读取麦克风设备…";
  listenStatus.dataset.state = "loading";
  refreshDevicesButton.disabled = true;
  listenDeviceSelect.disabled = true;

  try {
    const response = await fetch("/api/listen/devices");
    if (!response.ok) {
      throw new Error(`服务器返回错误状态：${response.status}`);
    }

    const result = await response.json();
    listenDeviceSelect.replaceChildren();
    listenUi.devicesLoaded = false;

    if (result.available !== true) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "音频后端不可用";
      listenDeviceSelect.append(option);
      setDeviceUsable(false);
      listenStatus.textContent =
        `无法枚举麦克风：${result.detail || "音频后端不可用"}。`
        + "请用带 numpy/sounddevice 的解释器启动后端。";
      listenStatus.dataset.state = "error";
      return;
    }

    const usable = (result.devices || [])
      .filter((device) => device.excluded !== true)
      .sort((a, b) => deviceRank(a) - deviceRank(b) || a.index - b.index);

    if (usable.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "没有可用的麦克风";
      listenDeviceSelect.append(option);
      setDeviceUsable(false);
      listenStatus.textContent =
        result.detail || "所有输入设备都被排除（虚拟/回环设备或名称不可靠）。";
      listenStatus.dataset.state = "error";
      return;
    }

    for (const device of usable) {
      const option = document.createElement("option");
      option.value = String(device.index);
      option.textContent = describeDevice(device) + (deviceRank(device) <= 1 ? " · 推荐" : "");
      listenDeviceSelect.append(option);
    }

    // 默认选排序后的第一个（MME 优先），索引全部来自后端，不硬编码。
    listenDeviceSelect.value = String(usable[0].index);
    listenDeviceSelect.disabled = false;
    listenUi.devicesLoaded = true;
    setDeviceUsable(true);
    listenStatus.textContent = `已找到 ${usable.length} 个可用麦克风，默认使用 ${usable[0].name}。`;
    listenStatus.dataset.state = "success";
  } catch (error) {
    listenDeviceSelect.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "读取设备失败";
    listenDeviceSelect.append(option);
    setDeviceUsable(false);
    listenStatus.textContent = `读取麦克风设备失败：${error.message}`;
    listenStatus.dataset.state = "error";
  } finally {
    refreshDevicesButton.disabled = false;
    renderListenButtons();
  }
}

// --------------------------------------------------------------------------
// 监听状态渲染与轮询
// --------------------------------------------------------------------------
function renderListenButtons() {
  const state = listenUi.state;
  const busy = state === "loading" || state === "listening" || state === "stopping";
  const startable = (state === "idle" || state === "error")
    && listenUi.devicesLoaded
    && listenUi.deviceUsable
    && listenDeviceSelect.value !== "";

  listenStartButton.disabled = !startable;
  listenStopButton.disabled = !(state === "loading" || state === "listening" || state === "error");
  refreshDevicesButton.disabled = busy;
  // 异常状态下允许改选设备后重试，否则设备一旦出错就没有退路。
  listenDeviceSelect.disabled =
    !(state === "idle" || state === "error") || !listenUi.devicesLoaded;
}

function renderListenState(state) {
  const normalized = LISTEN_STATE_TEXT[state] ? state : "error";
  listenUi.state = normalized;
  listenStateBadge.textContent = LISTEN_STATE_TEXT[normalized];
  listenStateBadge.dataset.state = normalized;
  renderListenButtons();
}

function renderListenMeta(status) {
  const device = status && status.device ? status.device : null;
  const frames = status && status.frames ? status.frames : {};
  const acoustic = status && status.acoustic ? status.acoustic : {};

  if (!status || listenUi.state === "idle") {
    listenMeta.textContent = "尚未开始监听。";
    return;
  }

  if (listenUi.state === "loading") {
    listenMeta.textContent = "正在装载识别模型并打开音频流，通常需要数秒，请稍候。";
    return;
  }

  if (listenUi.state === "error") {
    listenMeta.textContent = listenUi.lastError
      ? `异常原因：${listenUi.lastError}`
      : "监听异常，可重新开始监听。";
    return;
  }

  const parts = [];
  if (device && device.name && device.name !== "(未启动)") {
    parts.push(`设备：${device.name}`);
  }
  if (typeof status.listening_s === "number") {
    parts.push(`已监听 ${formatDuration(status.listening_s)}`);
  }
  if (typeof frames.scored === "number") {
    parts.push(`已评分帧 ${frames.scored}`);
  }
  if (typeof acoustic.persisted_events === "number") {
    parts.push(`本会话已落库 ${acoustic.persisted_events} 条`);
  }
  if (acoustic.persist_errors > 0) {
    parts.push(`落库失败 ${acoustic.persist_errors} 次`);
  }
  listenMeta.textContent = parts.join(" · ") || "正在监听。";
}

// 落库状态只能读 status.acoustic.knock_persist_enabled。
function renderPersistStatus() {
  if (listenUi.persistEnabled === null) {
    listenPersistStatus.textContent = "敲门事件落库状态未知：尚未读取到后端状态。";
    listenPersistStatus.dataset.state = "error";
    return;
  }

  if (listenUi.persistEnabled) {
    listenPersistStatus.textContent = "敲门事件落库已启用：识别到的敲门会写入事件历史。";
    listenPersistStatus.dataset.state = "success";
  } else {
    listenPersistStatus.textContent =
      "敲门事件落库未启用：可以继续监听，但识别结果不会写入事件历史（仅本会话计数）。"
      + "正式运行时需在启动进程前设置 SMARTLIFE_ENABLE_KNOCK_PERSIST=1。";
    listenPersistStatus.dataset.state = "error";
  }
}

function renderKnockAlertHint() {
  if (!listenUi.knockEnabledKnown) {
    knockAlertHint.textContent = "";
    return;
  }
  knockAlertHint.textContent = listenUi.knockEnabled
    ? "敲门提醒横幅：已开启。检测到新的声学敲门事件时会弹出醒目提醒。"
    : "敲门提醒横幅：已关闭。敲门事件仍会保存并显示在历史记录中，但不会弹出醒目提醒。";
}

function applyListenStatus(status) {
  if (!status || typeof status !== "object") {
    return;
  }

  const acoustic = status.acoustic || {};
  if (typeof acoustic.knock_persist_enabled === "boolean") {
    listenUi.persistEnabled = acoustic.knock_persist_enabled;
  }
  listenUi.persistedEvents = Number(acoustic.persisted_events) || 0;
  listenUi.persistErrors = Number(acoustic.persist_errors) || 0;
  listenUi.sessionId = status.session_id || null;
  listenUi.lastError = status.last_error || null;

  renderPersistStatus();
  renderListenState(status.state);
  renderListenMeta(status);
}

async function refreshStatusOnce(options = {}) {
  try {
    const response = await fetch("/api/listen/status");
    if (!response.ok) {
      throw new Error(`服务器返回错误状态：${response.status}`);
    }
    const status = await response.json();
    applyListenStatus(status);
    return status;
  } catch (error) {
    if (options.initial) {
      listenPersistStatus.textContent = `无法读取落库状态：${error.message}`;
      listenPersistStatus.dataset.state = "error";
    } else {
      listenStatus.textContent = `读取监听状态失败：${error.message}`;
      listenStatus.dataset.state = "error";
    }
    return null;
  }
}

function stopStatusPolling() {
  if (listenUi.statusTimer !== null) {
    window.clearInterval(listenUi.statusTimer);
    listenUi.statusTimer = null;
  }
}

function stopEventsPolling() {
  if (listenUi.eventsTimer !== null) {
    window.clearInterval(listenUi.eventsTimer);
    listenUi.eventsTimer = null;
  }
}

async function runStatusPoll() {
  const status = await refreshStatusOnce();
  if (!status) {
    return;
  }

  if (status.state === "listening") {
    startEventsPolling();
  } else if (status.state === "idle" || status.state === "error") {
    // 会话结束（或线程已退出）：停止轮询，避免空转。
    stopStatusPolling();
    stopEventsPolling();
    if (status.state === "idle") {
      listenStatus.textContent = "已停止监听。";
      listenStatus.dataset.state = "success";
    } else {
      listenStatus.textContent =
        `监听异常：${status.last_error || "未知原因"}。可重新开始监听。`;
      listenStatus.dataset.state = "error";
    }
  }
}

function startStatusPolling() {
  stopStatusPolling();
  listenUi.statusTimer = window.setInterval(runStatusPoll, STATUS_POLL_MS);
}

function startEventsPolling() {
  if (listenUi.eventsTimer !== null) {
    return;
  }
  listenUi.eventsTimer = window.setInterval(pollEvents, EVENTS_POLL_MS);
}

// --------------------------------------------------------------------------
// 开始 / 停止监听
// --------------------------------------------------------------------------
async function startListening() {
  const raw = listenDeviceSelect.value;
  if (raw === "" || raw === null) {
    listenStatus.textContent = "请先选择一个麦克风设备。";
    listenStatus.dataset.state = "error";
    return;
  }

  // 必须显式传 device，绝不依赖后端 target（可能落到不供数的端点）。
  const device = Number(raw);
  listenStartButton.disabled = true;
  listenStopButton.disabled = true;
  refreshDevicesButton.disabled = true;
  listenStatus.textContent = `正在请求启动监听（设备 #${device}）…`;
  listenStatus.dataset.state = "loading";

  try {
    const response = await fetch("/api/listen/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "microphone", device }),
    });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(extractErrorDetail(payload, response.status));
    }

    applyListenStatus(payload);
    // 202 只代表"已受理"：继续轮询，直到后端真正进入 listening。
    startStatusPolling();
    listenStatus.textContent = "启动请求已受理，正在等待后端进入监听状态…";
    listenStatus.dataset.state = "loading";
    await runStatusPoll();
  } catch (error) {
    listenStatus.textContent = `启动监听失败：${error.message}`;
    listenStatus.dataset.state = "error";
    renderListenState("error");
    listenUi.lastError = error.message;
    renderListenMeta({ state: "error" });
  } finally {
    renderListenButtons();
  }
}

async function stopListening() {
  listenStopButton.disabled = true;
  listenStartButton.disabled = true;
  refreshDevicesButton.disabled = true;
  // 先本地切到 stopping，再等后端确认，避免按钮与文案滞后。
  renderListenState("stopping");
  listenStatus.textContent = "正在停止监听…";
  listenStatus.dataset.state = "loading";

  try {
    const response = await fetch("/api/listen/stop", { method: "POST" });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(extractErrorDetail(payload, response.status));
    }

    applyListenStatus(payload);

    if (payload && payload.state === "idle") {
      stopStatusPolling();
      stopEventsPolling();
      listenStatus.textContent = "已停止监听。";
      listenStatus.dataset.state = "success";
      // 停止后再拉一次历史，补上最后写入的事件（不弹提醒）。
      await refreshEventListQuietly();
    } else if (payload && payload.state === "error") {
      stopStatusPolling();
      stopEventsPolling();
      listenStatus.textContent =
        `监听异常：${payload.last_error || "未知原因"}。可重新开始监听。`;
      listenStatus.dataset.state = "error";
    } else {
      // 仍是 stopping：继续轮询直到回到 idle。
      startStatusPolling();
    }
  } catch (error) {
    listenStatus.textContent = `停止监听失败：${error.message}`;
    listenStatus.dataset.state = "error";
    startStatusPolling();
  } finally {
    renderListenButtons();
  }
}

// --------------------------------------------------------------------------
// 事件轮询与敲门提醒
// --------------------------------------------------------------------------
async function refreshEventListQuietly() {
  try {
    const response = await fetch("/api/events?limit=50");
    if (!response.ok) {
      return;
    }
    const events = await response.json();
    renderEventList(events);
    // 会话已结束：这些事件属于刚结束的会话，只入历史、不再弹提醒。
    markEventsSeen(events);
    eventsStatus.textContent = `已加载 ${events.length} 条记录。`;
    eventsStatus.dataset.state = "success";
  } catch (error) {
    // 静默失败：不改变监听状态，也不打断提醒。
  }
}

async function pollEvents() {
  let events = null;
  try {
    const response = await fetch("/api/events?limit=50");
    if (!response.ok) {
      return;
    }
    events = await response.json();
  } catch (error) {
    return;
  }

  if (!Array.isArray(events) || events.length === 0) {
    return;
  }

  handleIncomingEvents(events);
}

function handleIncomingEvents(events) {
  const maxId = events.reduce(
    (max, event) => Math.max(max, Number(event.id) || 0),
    0,
  );

  // 基线尚未建立时（例如首屏读取失败），先补建基线，绝不补弹历史。
  if (!listenUi.baselineReady) {
    listenUi.lastEventId = maxId;
    listenUi.baselineReady = true;
    renderEventList(events);
    return;
  }

  const fresh = events.filter((event) => (Number(event.id) || 0) > listenUi.lastEventId);
  if (fresh.length === 0) {
    return;
  }

  listenUi.lastEventId = maxId;
  renderEventList(events);
  eventsStatus.textContent = `已加载 ${events.length} 条记录（新增 ${fresh.length} 条）。`;
  eventsStatus.dataset.state = "success";

  const knocks = fresh
    .filter((event) => event.source === "acoustic" && event.event_type === "knock")
    .sort((a, b) => (Number(a.id) || 0) - (Number(b.id) || 0));

  if (knocks.length === 0) {
    return;
  }

  let alertCount = 0;
  let latest = null;
  for (const event of knocks) {
    const id = Number(event.id);
    if (listenUi.alertedIds.has(id)) {
      continue;
    }
    listenUi.alertedIds.add(id);
    alertCount += 1;
    latest = event;
  }

  if (alertCount === 0 || latest === null) {
    return;
  }

  // 开关关闭：事件照常保存、历史照常显示，但不弹醒目横幅。
  if (!listenUi.knockEnabled) {
    return;
  }

  showKnockAlert(latest, alertCount);
}

function showKnockAlert(event, count) {
  const timeText = formatEventTime(event.occurred_at);
  const suffix = count > 1 ? `（本次共 ${count} 次，最近一次 #${event.id}）` : `（#${event.id}）`;
  knockAlertTime.textContent = `事件时间：${timeText} ${suffix}`;
  knockAlert.hidden = false;
  knockAlert.dataset.state = "visible";
  if (typeof knockAlert.scrollIntoView === "function") {
    knockAlert.scrollIntoView({ block: "nearest" });
  }
}

function hideKnockAlert() {
  knockAlert.hidden = true;
  knockAlert.dataset.state = "hidden";
}

refreshButton.addEventListener("click", checkBackend);
testAlertButton.addEventListener("click", sendTestAlert);
simulateEventForm.addEventListener("submit", createSimulatedEvent);
refreshEventsButton.addEventListener("click", loadEvents);
refreshDevicesButton.addEventListener("click", loadDevices);
listenStartButton.addEventListener("click", startListening);
listenStopButton.addEventListener("click", stopListening);
knockAlertDismiss.addEventListener("click", hideKnockAlert);
listenDeviceSelect.addEventListener("change", () => {
  setDeviceUsable(listenDeviceSelect.value !== "");
});

async function initializePage() {
  renderListenState("idle");
  renderPersistStatus();
  await Promise.all([checkBackend(), loadDevices(), refreshStatusOnce({ initial: true })]);
  renderKnockAlertHint();
}

initializePage();
