/**
 * light_controller.test.js — 光效逻辑验收测试
 * 运行： node src/light_controller.test.js
 * 用例与 docs/light_logic.md 第 9 节一一对应。
 */
import assert from "node:assert/strict";
import { LightController } from "./light_controller.js";

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); console.log(`  ✓ ${name}`); pass++; }
  catch (e) { console.log(`  ✗ ${name}\n      ${e.message}`); fail++; }
}

/** 稳定投喂 n 个相同样本，返回最后一次状态 */
function feed(lc, leq, peak, n, t0) {
  let s = null;
  for (let i = 0; i < n; i++) s = lc.push(leq, peak, t0 + i * 200);
  return s;
}

console.log("\n声学超标光效逻辑 · 验收测试\n");

test("1 安静：Δ=-8 → L0", () => {
  const lc = new LightController({ limit: 50 });
  const s = feed(lc, 42, 48, 6, 0);
  assert.equal(s.level, 0);
  assert.equal(s.mode, "steady");
});

test("2 轻微超标：Δ=+3 → L1", () => {
  const lc = new LightController({ limit: 50 });
  assert.equal(feed(lc, 53, 56, 6, 0).level, 1);
});

test("3 明显超标：Δ=+8 → L2", () => {
  const lc = new LightController({ limit: 50 });
  assert.equal(feed(lc, 58, 62, 6, 0).level, 2);
});

test("4 严重超标：Δ=+13 → L3", () => {
  const lc = new LightController({ limit: 50 });
  const s = feed(lc, 63, 66, 6, 0);
  assert.equal(s.level, 3);
  assert.equal(s.mode, "pulse");
});

test("5 极重超标：Δ=+20 → L4", () => {
  const lc = new LightController({ limit: 50 });
  const s = feed(lc, 70, 76, 6, 0);
  assert.equal(s.level, 4);
  assert.equal(s.mode, "strobe");
  assert.ok(s.brightness >= 0.99, "L4 亮度应接近 100%");
});

test("6 峰值冲击：Leq 达标但 Peak 超 26 dB → 强制升一级", () => {
  const lc = new LightController({ limit: 50 });
  const s = feed(lc, 48, 76, 6, 0);
  assert.equal(s.level, 1, "应由 L0 升至 L1");
});

test("7 临界抖动：在 49/51 之间反复跳变，状态迁移 ≤ 1 次", () => {
  const lc = new LightController({ limit: 50 });
  let t = 0;
  feed(lc, 49, 52, 6, t); t += 1200;
  const before = lc.events.length;
  for (let i = 0; i < 25; i++) {           // 5 s 抖动
    lc.push(i % 2 ? 51 : 49, 53, t + i * 200);
  }
  const after = lc.events.length;
  assert.ok(after - before <= 1, `发生了 ${after - before} 次迁移`);
});

test("8 回差生效：从 L2 回落需连续 3 s 才降级", () => {
  const lc = new LightController({ limit: 50 });
  let t = 0;
  feed(lc, 58, 62, 6, t); t += 1200;       // 进入 L2
  assert.equal(lc.cur, 2);

  // 投喂 Δ=-1（49 dB）：落在回差带内（> 5-3=2 才降），不应立刻降
  let s;
  for (let i = 0; i < 15; i++) s = lc.push(49, 52, t + i * 200);   // 到 2.8 s
  assert.equal(s.level, 2, "2.8 s 时仍应为 L2");

  for (let i = 15; i < 20; i++) s = lc.push(49, 52, t + i * 200);  // 到 3.8 s
  assert.equal(s.level, 0, "超过 3 s 后应降级");
});

test("9 升级零延迟：Δ 由 -2 跳到 +7，立即 L2（无 3 s 等待）", () => {
  // smoothN:1 隔离状态机本身，排除 1 s 能量平均的平滑影响
  const lc = new LightController({ limit: 50, smoothN: 1 });
  feed(lc, 48, 51, 6, 0);
  const s = lc.push(57, 60, 1200);
  assert.equal(s.level, 2);
  assert.equal(s.changed, true);
  assert.equal(s.dir, 1);
});

test("10 夜间模式：Δ=+20 最高只到 L3，且不爆闪", () => {
  const lc = new LightController({ limit: 45, night: true });
  const s = feed(lc, 65, 80, 6, 0);
  assert.equal(s.level, 3);
  assert.equal(s.mode, "pulse");
  assert.ok(s.brightness <= 0.4501, `夜间亮度应 ≤45%，实为 ${s.brightness}`);
});

test("11 能量平均正确性：[80,80,80,40,40] → 77.8 dB（非算术平均 64）", () => {
  const v = LightController.leq([80, 80, 80, 40, 40]);
  assert.ok(Math.abs(v - 77.78) < 0.05, `实为 ${v.toFixed(2)}`);
});

test("12 亮度随偏离量单调递增", () => {
  const lc = new LightController({ limit: 50 });
  const b = [2, 6, 10, 15, 20].map(d => lc.brightness(lc.rawLevel(d), d));
  for (let i = 1; i < b.length; i++) assert.ok(b[i] >= b[i - 1], "亮度应非递减");
});

console.log(`\n结果：${pass} 通过 / ${fail} 失败\n`);
process.exit(fail ? 1 : 0);
