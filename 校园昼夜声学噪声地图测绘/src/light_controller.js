/**
 * light_controller.js — 声学超标自动提醒光效逻辑 · 参考实现
 * ICEC-2026-0139 · 课题 06 校园昼夜声学噪声地图测绘
 *
 * 设计说明见 docs/light_logic.md。本文件不依赖任何框架/硬件，可直接移植到
 * ESP32（替换 output() 为 PWM 驱动）或在浏览器中运行。
 *
 * 用法：
 *   const lc = new LightController({ limit: 50 });            // 图书馆昼间
 *   const s  = lc.push(leq1s, peak1s, tMs);                   // 每 200 ms 调用一次
 *   rgbWrite(s.color, s.brightness);                          // 交给灯带
 */

export const LEVELS = [
  { level: 0, key: "L0", name: "达标", color: "#16a34a", mode: "steady", freq: 0.0, duty: 1.0 },
  { level: 1, key: "L1", name: "轻微超标", color: "#eab308", mode: "breathe", freq: 0.25, duty: 1.0 },
  { level: 2, key: "L2", name: "明显超标", color: "#f97316", mode: "breathe", freq: 0.6, duty: 1.0 },
  { level: 3, key: "L3", name: "严重超标", color: "#dc2626", mode: "pulse", freq: 1.2, duty: 0.5 },
  { level: 4, key: "L4", name: "极重超标", color: "#b91c1c", mode: "strobe", freq: 2.0, duty: 0.3, alt: "#ffffff" },
];

const DEFAULT = {
  bands: [0, 5, 10, 15],   // 分级边界（Δ，dB）
  hysteresis: 3,           // 回差 dB
  holdMs: 3000,            // 降级前需连续低于回落线的时间
  dwellMs: 2000,           // 最小驻留
  peakBoost: 20,           // Peak 超出限值多少 → 强制升一级
  nightFactor: 0.45,       // 夜间亮度系数
  nightMaxLevel: 3,        // 夜间允许的最高等级（禁爆闪）
  smoothN: 5,              // 1 s 能量平均样本数（5 × 200 ms）
};

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export class LightController {
  /**
   * @param {object} opt
   * @param {number} opt.limit   该点位当前时段的功能区限值 dB(A)
   * @param {boolean} [opt.night] 是否夜间模式
   */
  constructor(opt = {}) {
    this.limit = opt.limit;
    if (typeof this.limit !== "number") throw new Error("必须提供 limit");
    this.cfg = { ...DEFAULT, ...opt };
    this.night = !!opt.night;
    this.buf = [];                 // 1 s 滑动窗
    this.cur = 0;
    this.cand = null;
    this.candT = 0;
    this.lastChange = -Infinity;
    this.events = [];
  }

  /** 夜间模式切换（限值同时按昼/夜切换时由调用方设置 limit） */
  setNight(night) { this.night = !!night; return this; }
  setLimit(limit) { this.limit = limit; return this; }

  /** 能量平均：必须在能量域求和，不能算术平均 */
  static leq(values) {
    if (!values.length) return 0;
    const e = values.reduce((s, v) => s + Math.pow(10, v / 10), 0) / values.length;
    return 10 * Math.log10(e);
  }

  /** 原始分级：Δ 落在哪个区间 */
  rawLevel(delta) {
    let lv = 0;
    for (const b of this.cfg.bands) if (delta > b) lv++;
    return Math.min(4, lv);
  }

  brightness(level, delta) {
    let b = level === 0 ? 0.20
      : Math.min(1, 0.35 + 0.65 * Math.min(1, Math.max(0, delta) / 20));
    if (this.night) b *= this.cfg.nightFactor;
    return Math.round(b * 1000) / 1000;
  }

  /**
   * 投喂一个 200 ms 样本。
   * @param {number} leq   该 200 ms 窗的 A 计权有效值 dB
   * @param {number} peak  该窗的瞬时峰值 dB
   * @param {number} tMs   单调时间戳（ms）
   */
  push(leq, peak, tMs) {
    this.buf.push(leq);
    if (this.buf.length > this.cfg.smoothN) this.buf.shift();
    const leq1s = this.buf.length === this.cfg.smoothN ? LightController.leq(this.buf) : leq;
    const delta = leq1s - this.limit;

    // ---- 目标等级：连续量取整 + 峰值加权 + 夜间限档 ----
    let target = this.rawLevel(delta);
    if (peak - this.limit >= this.cfg.peakBoost) target = Math.min(4, target + 1);
    if (this.night) target = Math.min(target, this.cfg.nightMaxLevel);

    // ---- 状态机：快升慢降 ----
    let changed = false, dir = 0;
    if (target > this.cur) {
      this.cur = target;                 // 升级零延迟
      this.cand = null;
      this.lastChange = tMs;
      changed = true; dir = 1;
    } else if (target < this.cur) {
      // 回差：必须低于「当前等级下界 − hysteresis」才允许降
      const lowerBound = this.cfg.bands[this.cur - 1];
      const passHysteresis = delta < lowerBound - this.cfg.hysteresis;
      if (passHysteresis) {
        if (this.cand !== target) { this.cand = target; this.candT = tMs; }
        const held = tMs - this.candT >= this.cfg.holdMs;
        const dwelled = tMs - this.lastChange >= this.cfg.dwellMs;
        if (held && dwelled) {
          this.cur = target; this.cand = null; this.lastChange = tMs;
          changed = true; dir = -1;
        }
      } else {
        this.cand = null;              // 仍在回差带内，保持原等级
      }
    } else {
      this.cand = null;
    }

    const L = LEVELS[this.cur];
    if (changed) this.events.push({ t: tMs, level: this.cur, delta: +delta.toFixed(1) });

    return {
      level: this.cur,
      key: L.key,
      name: L.name,
      color: L.color,
      alt: L.alt || null,
      mode: L.mode,
      freq: L.freq,
      duty: L.duty,
      brightness: this.brightness(this.cur, delta),
      leq1s: +leq1s.toFixed(1),
      delta: +delta.toFixed(1),
      changed,
      dir,                                  // 1 升 / -1 降 / 0 不变
      pendingDown: this.cand !== null,      // 正在等待降级
    };
  }
}

export default LightController;
