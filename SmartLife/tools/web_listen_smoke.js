/*
 * 网页端逻辑测试（无新依赖）：用 Node 内置 vm 加载 static/js/app.js，
 * 以最小 DOM 存根 + 假 fetch 驱动，不启动浏览器、不启动麦克风、不碰数据库。
 *
 * 运行： node tools/web_listen_smoke.js
 */

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const APP_JS = path.join(PROJECT_ROOT, "static", "js", "app.js");
const INDEX_HTML = path.join(PROJECT_ROOT, "static", "index.html");
const APP_CSS = path.join(PROJECT_ROOT, "static", "css", "styles.css");
const SOURCE = fs.readFileSync(APP_JS, "utf8");

let passed = 0;
const failures = [];

function check(name, condition, detail) {
  if (condition) {
    passed += 1;
    console.log(`  PASS  ${name}`);
  } else {
    failures.push(name);
    console.log(`  FAIL  ${name}${detail === undefined ? "" : `  → ${detail}`}`);
  }
}

function eq(name, actual, expected) {
  check(name, actual === expected, `期望 ${JSON.stringify(expected)}，实际 ${JSON.stringify(actual)}`);
}

function includes(name, haystack, needle) {
  const text = String(haystack ?? "");
  check(name, text.includes(needle), `「${text}」不含「${needle}」`);
}

// ---------------------------------------------------------------- DOM 存根
class El {
  constructor(selector) {
    this.selector = selector;
    this.tagName = "DIV";
    this._text = "";
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.checked = false;
    this.type = "";
    this.className = "";
    this.children = [];
    this.attrs = {};
    this.listeners = {};
  }

  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }

  get textContent() {
    if (this.children.length > 0) {
      return this.children.map((child) => child.textContent ?? "").join("");
    }
    return this._text;
  }

  append(...nodes) {
    for (const node of nodes) {
      this.children.push(node);
    }
    this._text = this.children.map((n) => n.textContent ?? "").join("");
  }

  replaceChildren(...nodes) {
    this.children = nodes;
    this._text = nodes.map((n) => n.textContent ?? "").join("");
  }

  setAttribute(key, value) {
    this.attrs[key] = String(value);
  }

  addEventListener(type, handler) {
    if (!this.listeners[type]) {
      this.listeners[type] = [];
    }
    this.listeners[type].push(handler);
  }

  async fire(type) {
    const handlers = this.listeners[type] || [];
    for (const handler of handlers) {
      await handler({ preventDefault() {} });
    }
  }

  scrollIntoView() {}
}

function createSandbox({ api }) {
  const elements = new Map();
  const intervals = new Map();
  let intervalSeq = 0;

  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) {
        const el = new El(selector);
        // HTML 里 <section id="knock-alert" ... hidden>，初始就是隐藏的。
        if (selector === "#knock-alert") {
          el.hidden = true;
        }
        elements.set(selector, el);
      }
      return elements.get(selector);
    },
    createElement(tag) {
      const el = new El(`<${tag}>`);
      el.tagName = String(tag).toUpperCase();
      return el;
    },
  };

  const window = {
    setInterval(fn, ms) {
      intervalSeq += 1;
      intervals.set(intervalSeq, { fn, ms });
      return intervalSeq;
    },
    clearInterval(id) {
      intervals.delete(id);
    },
  };

  class FormDataStub {
    constructor() {
      this.values = { event_type: "knock", confidence: "0.8" };
    }
    get(key) {
      return this.values[key];
    }
  }

  const sandbox = {
    document,
    window,
    fetch: (url, options) => api.handle(url, options),
    FormData: FormDataStub,
    console,
    Date,
    Math,
    JSON,
    Set,
    Map,
    Number,
    String,
    Object,
    Array,
    Promise,
    Error,
    encodeURIComponent,
  };
  sandbox.globalThis = sandbox;

  const context = vm.createContext(sandbox);
  vm.runInContext(SOURCE, context, { filename: "app.js" });

  const el = (selector) => document.querySelector(selector);
  const read = (expr) => vm.runInContext(expr, context);
  const intervalMs = () => Array.from(intervals.values()).map((item) => item.ms);

  return { context, elements: el, read, intervalMs, intervals, api };
}

function flush(times = 8) {
  let chain = Promise.resolve();
  for (let i = 0; i < times; i += 1) {
    chain = chain.then(() => new Promise((resolve) => setImmediate(resolve)));
  }
  return chain;
}

// ---------------------------------------------------------------- 假接口
function makeApi(overrides = {}) {
  const calls = [];
  const state = {
    health: {
      status: "ok",
      message: "SmartLife 后端已连接。",
      test_alert_available: true,
      listen_persist: false,
    },
    devices: {
      available: true,
      devices: [
        { index: 14, name: "麦克风阵列 (Intel SST)", hostapi: "Windows WASAPI", hostapi_index: 2, max_input_channels: 2, default_samplerate: 48000, supports_48k: true, excluded: false, exclude_reason: null },
        { index: 7, name: "麦克风阵列 (Intel SST)", hostapi: "Windows DirectSound", hostapi_index: 1, max_input_channels: 2, default_samplerate: 44100, supports_48k: false, excluded: false, exclude_reason: null },
        { index: 1, name: "麦克风阵列 (Intel SST)", hostapi: "MME", hostapi_index: 0, max_input_channels: 2, default_samplerate: 44100, supports_48k: false, excluded: false, exclude_reason: null },
        { index: 18, name: "扬声器回环", hostapi: "Windows WDM-KS", hostapi_index: 3, max_input_channels: 2, default_samplerate: 48000, supports_48k: true, excluded: true, exclude_reason: "宿主 API「Windows WDM-KS」的设备名不可靠" },
      ],
      target: null,
      detail: null,
      note: "只查询设备列表。",
    },
    status: {
      state: "idle",
      source: null,
      session_id: null,
      listening_s: 0,
      uptime_s: 0,
      frames: { produced: 0, scored: 0, warmup_dropped: 0 },
      device: { index: null, name: "(未启动)", hostapi: "none", sample_rate: 0, opened: false },
      acoustic: {
        knock_persist_enabled: false,
        persisted_events: 0,
        persist_errors: 0,
        persist_last_error: null,
        last_event_id: null,
      },
      last_error: null,
      stop_reason: null,
      events: [],
    },
    events: [
      { id: 1, event_type: "doorbell", occurred_at: "2026-11-15T03:00:51.572937Z", confidence: 0.9, source: "simulated" },
      { id: 2, event_type: "knock", occurred_at: "2026-11-15T03:01:10.000000Z", confidence: 0.8, source: "simulated" },
      { id: 3, event_type: "knock", occurred_at: "2026-11-15T03:02:00.000000Z", confidence: null, source: "acoustic" },
    ],
    alerts: [
      { event_type: "alarm", enabled: true, updated_at: "2026-11-15T03:00:00Z" },
      { event_type: "doorbell", enabled: true, updated_at: "2026-11-15T03:00:00Z" },
      { event_type: "kettle", enabled: true, updated_at: "2026-11-15T03:00:00Z" },
      { event_type: "knock", enabled: true, updated_at: "2026-11-15T03:00:00Z" },
    ],
    start: { status: 202, body: null },
    stop: { status: 200, body: null },
    ...overrides,
  };

  const json = (status, body) => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  });

  async function handle(url, options = {}) {
    const method = (options.method || "GET").toUpperCase();
    calls.push({ url, method, body: options.body || null });

    if (url === "/api/health") {
      return json(200, state.health);
    }
    if (url === "/api/listen/devices") {
      return json(200, state.devices);
    }
    if (url === "/api/listen/status") {
      return json(200, state.status);
    }
    if (url === "/api/listen/start") {
      const body = state.start.body || {
        ...state.status,
        state: "loading",
        source: "microphone",
        session_id: "abc123",
      };
      if (state.start.status < 300) {
        // 真实后端受理后会立刻转入 loading，后续 status 轮询读到的就是这个状态。
        state.status = { ...state.status, ...body };
      }
      return json(state.start.status, body);
    }
    if (url === "/api/listen/stop") {
      const body = state.stop.body || { ...state.status, state: "idle" };
      if (state.stop.status < 300) {
        state.status = { ...state.status, ...body };
      }
      return json(state.stop.status, body);
    }
    if (url.startsWith("/api/events?")) {
      return json(200, state.events);
    }
    if (url === "/api/events/simulate") {
      const created = { id: 99, event_type: "knock", occurred_at: "2026-11-18T04:00:00Z", confidence: 0.8, source: "simulated" };
      state.events = [...state.events, created];
      return json(201, created);
    }
    if (url === "/api/settings/alerts") {
      return json(200, state.alerts);
    }
    if (url.startsWith("/api/settings/alerts/")) {
      const type = decodeURIComponent(url.split("/").pop());
      const payload = JSON.parse(options.body || "{}");
      state.alerts = state.alerts.map((item) =>
        item.event_type === type ? { ...item, enabled: payload.enabled } : item,
      );
      const saved = state.alerts.find((item) => item.event_type === type);
      return json(200, saved);
    }
    if (url === "/api/test-alert") {
      return json(200, { kind: "test", message: "测试提醒", created_at: "2026-11-18T04:00:00Z" });
    }
    return json(404, { detail: `未模拟的地址：${url}` });
  }

  return { handle, state, calls };
}

async function boot(overrides) {
  const api = makeApi(overrides);
  const page = createSandbox({ api });
  await flush(10);
  return page;
}

// ---------------------------------------------------------------- 用例
async function testDevices() {
  console.log("\n[1] 麦克风设备选择");
  const page = await boot();
  const select = page.elements("#listen-device");

  eq("选项数量=未排除设备数(3)", select.children.length, 3);
  eq("默认选中 MME 设备 index=1", select.value, "1");
  eq("第一个选项就是默认项", select.children[0].value, "1");
  includes("默认项标记为推荐", select.children[0].textContent, "推荐");
  includes("默认项显示宿主 API", select.children[0].textContent, "MME");
  check(
    "被排除设备未进入下拉",
    !select.children.some((option) => option.value === "18"),
  );
  eq("设备可选中时开始按钮可用", page.elements("#listen-start-button").disabled, false);

  const page2 = await boot({
    devices: { available: true, devices: [{ index: 18, name: "回环", hostapi: "Windows WDM-KS", hostapi_index: 3, max_input_channels: 2, default_samplerate: 48000, supports_48k: true, excluded: true, exclude_reason: "名称不可靠" }], target: null, detail: "所有输入设备都被排除（虚拟设备或名称不可靠）。", note: "" },
  });
  eq("无可用设备：开始按钮禁用", page2.elements("#listen-start-button").disabled, true);
  includes("无可用设备：显示原因", page2.elements("#listen-status").textContent, "所有输入设备都被排除");

  const page3 = await boot({
    devices: { available: false, devices: [], target: null, detail: "音频后端不可用（无法导入 sounddevice）", note: "" },
  });
  eq("音频后端不可用：开始按钮禁用", page3.elements("#listen-start-button").disabled, true);
  includes("音频后端不可用：显示原因", page3.elements("#listen-status").textContent, "sounddevice");
}

async function testStartStop() {
  console.log("\n[2] 开始 / 停止与状态机");
  const page = await boot();
  const api = page.api;

  await page.elements("#listen-start-button").fire("click");
  await flush(10);

  const startCall = api.calls.find((call) => call.url === "/api/listen/start");
  const body = JSON.parse(startCall.body || "{}");
  eq("start 的 source 是 microphone", body.source, "microphone");
  eq("start 显式传 device=1", body.device, 1);
  check("start 请求体里 device 不是 null", body.device !== null && body.device !== undefined);
  eq("202 后状态显示启动中", page.elements("#listen-state").textContent, "启动中");
  check(
    "202 后不谎报正在监听",
    page.elements("#listen-state").textContent !== "正在监听",
  );
  eq("启动中：开始按钮禁用", page.elements("#listen-start-button").disabled, true);
  eq("启动中：停止按钮可用", page.elements("#listen-stop-button").disabled, false);
  check("已启动 1.5 秒状态轮询", page.intervalMs().includes(1500), page.intervalMs());

  // 后端真正进入 listening
  api.state.status = {
    ...api.state.status,
    state: "listening",
    source: "microphone",
    session_id: "abc123",
    listening_s: 12.4,
    frames: { produced: 40, scored: 36, warmup_dropped: 2 },
    device: { index: 1, name: "麦克风阵列 (Intel SST)", hostapi: "MME", sample_rate: 48000, opened: true },
  };
  await page.context.runStatusPoll();
  await flush(4);

  eq("状态变为正在监听", page.elements("#listen-state").textContent, "正在监听");
  includes("监听中显示设备名", page.elements("#listen-meta").textContent, "麦克风阵列");
  includes("监听中显示已评分帧", page.elements("#listen-meta").textContent, "36");
  check("已启动 2 秒事件轮询", page.intervalMs().includes(2000), page.intervalMs());

  await page.elements("#listen-stop-button").fire("click");
  await flush(10);
  eq("停止后回到未监听", page.elements("#listen-state").textContent, "未监听");
  eq("停止后：停止按钮禁用", page.elements("#listen-stop-button").disabled, true);
  eq("停止后：开始按钮恢复", page.elements("#listen-start-button").disabled, false);

  const stopMs = page.intervalMs();
  check("停止后清掉事件轮询", !stopMs.includes(2000), stopMs);
}

async function testStateMapping() {
  console.log("\n[3] 五种状态文案与异常信息");
  const page = await boot();
  const cases = [
    ["idle", "未监听"],
    ["loading", "启动中"],
    ["listening", "正在监听"],
    ["stopping", "停止中"],
    ["error", "异常"],
  ];
  for (const [state, text] of cases) {
    page.context.applyListenStatus({ ...page.api.state.status, state });
    eq(`state=${state} 显示「${text}」`, page.elements("#listen-state").textContent, text);
    eq(`state=${state} 徽标 data-state`, page.elements("#listen-state").dataset.state, state);
  }

  page.context.applyListenStatus({
    ...page.api.state.status,
    state: "error",
    last_error: "AudioDeviceLost: 音频输入流已停止",
  });
  includes("异常显示后端原因", page.elements("#listen-meta").textContent, "音频输入流已停止");
}

async function testStartErrors() {
  console.log("\n[4] 启动失败的错误语义");

  const page400 = await boot({
    start: { status: 400, body: { detail: "设备 18「回环」已被排除：名称不可靠" } },
  });
  await page400.elements("#listen-start-button").fire("click");
  await flush(10);
  includes("400 显示 detail 文本", page400.elements("#listen-status").textContent, "已被排除");
  eq("400 后状态为异常", page400.elements("#listen-state").textContent, "异常");
  eq("400 后开始按钮可重试", page400.elements("#listen-start-button").disabled, false);
  eq("400 后可改选设备重试", page400.elements("#listen-device").disabled, false);

  const page409 = await boot({
    start: { status: 409, body: { detail: { message: "当前状态为 loading，不能重复启动监听。", state: "loading" } } },
  });
  await page409.elements("#listen-start-button").fire("click");
  await flush(10);
  includes("409 显示 message", page409.elements("#listen-status").textContent, "不能重复启动监听");
  includes("409 显示当前状态", page409.elements("#listen-status").textContent, "loading");

  const page503 = await boot({
    start: { status: 503, body: { detail: "识别栈不可用：缺少 numpy" } },
  });
  await page503.elements("#listen-start-button").fire("click");
  await flush(10);
  includes("503 显示后端原因", page503.elements("#listen-status").textContent, "缺少 numpy");
}

async function testHistoryBaseline() {
  console.log("\n[5] 历史基线与刷新不重复提醒");
  const page = await boot();
  eq("首屏不弹提醒", page.elements("#knock-alert").hidden, true);
  eq("首屏基线=最大历史 id", page.read("listenUi.lastEventId"), 3);
  eq("首屏历史渲染 3 条", page.elements("#event-list").children.length, 3);

  // 刷新页面（重新加载脚本）后，历史里仍有 acoustic knock(id=3)
  const pageAgain = await boot();
  eq("刷新后仍不弹提醒", pageAgain.elements("#knock-alert").hidden, true);
  eq("刷新后基线仍是 3", pageAgain.read("listenUi.lastEventId"), 3);
}

async function testNewEventAlert() {
  console.log("\n[6] 新事件提醒、同 ID 去重、开关关闭");
  const page = await boot();
  const api = page.api;

  api.state.events = [
    { id: 4, event_type: "knock", occurred_at: "2026-11-18T04:10:00.000000Z", confidence: null, source: "acoustic" },
    ...api.state.events,
  ];
  await page.context.pollEvents();
  await flush(6);

  eq("新 acoustic knock 弹出提醒", page.elements("#knock-alert").hidden, false);
  // 提醒标题是 HTML 里的静态文案，存根不解析标记，因此对源文件断言。
  const indexHtml = fs.readFileSync(INDEX_HTML, "utf8");
  includes("提醒文案为「检测到可能的敲门声」", indexHtml, "检测到可能的敲门声");
  check("横幅是 role=alert 且 assertive", /id="knock-alert"[\s\S]{0,200}?role="alert"/.test(indexHtml) && indexHtml.includes('aria-live="assertive"'));
  includes("提醒显示事件时间", page.elements("#knock-alert-time").textContent, "事件时间：");
  includes("提醒带事件编号", page.elements("#knock-alert-time").textContent, "#4");
  eq("历史列表更新为 4 条", page.elements("#event-list").children.length, 4);

  // 点"我知道了"
  await page.elements("#knock-alert-dismiss").fire("click");
  eq("我知道了 可关闭提醒", page.elements("#knock-alert").hidden, true);

  // 同一批事件再轮询：不得重复提醒
  await page.context.pollEvents();
  await flush(6);
  eq("同一 ID 不重复提醒", page.elements("#knock-alert").hidden, true);
  eq("已提醒集合记录 id=4", page.read("listenUi.alertedIds.has(4)"), true);

  // 非 acoustic 事件不提醒
  api.state.events = [
    { id: 5, event_type: "knock", occurred_at: "2026-11-18T04:11:00.000000Z", confidence: 0.8, source: "simulated" },
    ...api.state.events,
  ];
  await page.context.pollEvents();
  await flush(6);
  eq("模拟来源的敲门不弹提醒", page.elements("#knock-alert").hidden, true);

  // acoustic 但非 knock 不提醒
  api.state.events = [
    { id: 6, event_type: "doorbell", occurred_at: "2026-11-18T04:12:00.000000Z", confidence: null, source: "acoustic" },
    ...api.state.events,
  ];
  await page.context.pollEvents();
  await flush(6);
  eq("acoustic 门铃不弹敲门提醒", page.elements("#knock-alert").hidden, true);

  // 关闭 knock 开关
  const toggle = page.elements("#settings-list").children
    .find((row) => row.children.some((child) => String(child.textContent).includes("敲门")));
  check("设置里能找到敲门开关", Boolean(toggle));
  const checkbox = toggle.children[toggle.children.length - 1];
  checkbox.checked = false;
  await checkbox.fire("change");
  await flush(8);

  eq("关闭开关后不显示横幅", page.elements("#knock-alert").hidden, true);
  includes("设置提示说明仍会保存", page.elements("#knock-alert-hint").textContent, "仍会保存");

  api.state.events = [
    { id: 7, event_type: "knock", occurred_at: "2026-11-18T04:13:00.000000Z", confidence: null, source: "acoustic" },
    ...api.state.events,
  ];
  await page.context.pollEvents();
  await flush(6);
  eq("开关关闭时新事件不弹横幅", page.elements("#knock-alert").hidden, true);
  eq("开关关闭时事件仍进历史", page.elements("#event-list").children.length, 7);
}

async function testPersistStatus() {
  console.log("\n[7] 落库状态显示");

  const off = await boot();
  includes("默认未启用文案", off.elements("#listen-persist-status").textContent, "敲门事件落库未启用");
  includes("未启用时提示不会写入历史", off.elements("#listen-persist-status").textContent, "不会写入事件历史");
  eq("未启用时开始按钮仍可用", off.elements("#listen-start-button").disabled, false);

  const on = await boot({
    status: {
      state: "idle",
      source: null,
      session_id: null,
      listening_s: 0,
      uptime_s: 0,
      frames: { produced: 0, scored: 0, warmup_dropped: 0 },
      device: { index: null, name: "(未启动)", hostapi: "none", sample_rate: 0, opened: false },
      acoustic: { knock_persist_enabled: true, persisted_events: 2, persist_errors: 0, persist_last_error: null, last_event_id: 12 },
      last_error: null,
      stop_reason: null,
      events: [],
    },
  });
  includes("启用文案", on.elements("#listen-persist-status").textContent, "敲门事件落库已启用");
  eq("启用时标记 success", on.elements("#listen-persist-status").dataset.state, "success");

  // health.listen_persist=false 不得影响落库显示
  includes(
    "不拿 health.listen_persist 冒充落库开关",
    on.elements("#listen-persist-status").textContent,
    "已启用",
  );
}

async function testLegacyFeatures() {
  console.log("\n[8] 原有功能未被破坏");
  const page = await boot();
  const api = page.api;

  includes("连接状态显示后端消息", page.elements("#connection-status").textContent, "SmartLife 后端已连接");
  eq("连接状态标记 success", page.elements("#connection-status").dataset.state, "success");

  eq("提醒设置渲染 4 行", page.elements("#settings-list").children.length, 4);

  // 模拟事件
  await page.elements("#simulate-event-form").fire("submit");
  await flush(10);
  const simulateCall = api.calls.find((call) => call.url === "/api/events/simulate");
  check("模拟事件仍会 POST", Boolean(simulateCall));
  includes(
    "模拟事件成功提示仍在",
    page.elements("#simulate-event-status").textContent,
    "已写入数据库",
  );
  eq("模拟事件后列表刷新", page.elements("#event-list").children.length, 4);

  // 测试提醒
  await page.elements("#test-alert-button").fire("click");
  await flush(8);
  includes("测试提醒仍可用", page.elements("#test-alert-status").textContent, "测试提醒");

  // 提醒设置 PATCH 成功
  const doorbellRow = page.elements("#settings-list").children[0];
  const doorbellToggle = doorbellRow.children[doorbellRow.children.length - 1];
  doorbellToggle.checked = false;
  await doorbellToggle.fire("change");
  await flush(8);
  includes("提醒设置保存成功", page.elements("#settings-status").textContent, "已关闭并保存");

  // 原有的"刷新记录"按钮仍工作
  await page.elements("#refresh-events-button").fire("click");
  await flush(8);
  includes("刷新记录仍工作", page.elements("#events-status").textContent, "已加载");
}

async function testEventRenderAndFailure() {
  console.log("\n[9] 历史渲染与连接失败");
  const page = await boot();
  const items = page.elements("#event-list").children.map((item) => item.textContent);
  includes("历史显示模拟来源标记", items.join("\n"), "模拟事件");
  includes("历史显示 acoustic 来源", items.join("\n"), "acoustic");
  includes("acoustic 事件置信度显示未提供", items.join("\n"), "置信度未提供");

  const api = makeApi();
  api.handle = async (url) => {
    if (url === "/api/health") {
      return { ok: false, status: 500, json: async () => ({}) };
    }
    return makeApi().handle(url);
  };
  const broken = createSandbox({ api });
  await flush(10);
  includes("连接失败提示仍然出现", broken.elements("#connection-status").textContent, "连接失败");
}

async function testStaticAssets() {
  console.log("\n[10] 静态资源与选择器一致性");
  const html = fs.readFileSync(INDEX_HTML, "utf8");
  const css = fs.readFileSync(APP_CSS, "utf8");

  const requiredIds = [
    "knock-alert", "knock-alert-time", "knock-alert-dismiss",
    "listen-device", "listen-start-button", "listen-stop-button",
    "refresh-devices-button", "listen-state", "listen-meta",
    "listen-persist-status", "listen-status", "knock-alert-hint",
    "connection-status", "refresh-button", "test-alert-card",
    "test-alert-button", "test-alert-status", "simulate-event-form",
    "event-type", "event-confidence", "simulate-event-button",
    "simulate-event-status", "settings-list", "settings-status",
    "event-list", "events-status", "refresh-events-button",
  ];
  for (const id of requiredIds) {
    check(`HTML 含 id="${id}"`, html.includes(`id="${id}"`));
  }

  const selectors = Array.from(SOURCE.matchAll(/document\.querySelector\("#([a-z-]+)"\)/g)).map((m) => m[1]);
  const missing = selectors.filter((id) => !html.includes(`id="${id}"`));
  check("JS 选择器全部能在 HTML 找到", missing.length === 0, `缺失：${missing.join(", ")}`);

  check("HTML 引用了 app.js 0.4.0", html.includes("app.js?v=0.4.0"));
  check("CSS 含状态徽标样式", css.includes(".state-badge"));
  check("CSS 含提醒横幅样式", css.includes(".alert-banner"));
  check("CSS 含减少动效降级", css.includes("prefers-reduced-motion"));
  check("JS 不含 navigator.vibrate", !SOURCE.includes("navigator.vibrate"));
  check("JS 未引用 health.listen_persist", !SOURCE.includes("listen_persist"));
}

async function main() {
  console.log("网页端逻辑测试（假接口 + DOM 存根，无浏览器、无麦克风、无数据库）");
  await testDevices();
  await testStartStop();
  await testStateMapping();
  await testStartErrors();
  await testHistoryBaseline();
  await testNewEventAlert();
  await testPersistStatus();
  await testLegacyFeatures();
  await testEventRenderAndFailure();
  await testStaticAssets();

  console.log(`\n结果：${passed} PASS / ${failures.length} FAIL`);
  if (failures.length > 0) {
    console.log("失败项：");
    for (const name of failures) {
      console.log(`  - ${name}`);
    }
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error("测试脚本异常：", error);
  process.exitCode = 1;
});
