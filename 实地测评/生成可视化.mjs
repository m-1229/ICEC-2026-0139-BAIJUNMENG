import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const outputDirectory = path.dirname(fileURLToPath(import.meta.url));
const chartDirectory = path.join(outputDirectory, "可视化图");
const gravity = 9.81;
const samples = [
  { file: "静止.csv", label: "静止" },
  { file: "平地慢走.csv", label: "平地慢走" },
  { file: "上楼.csv", label: "上楼" },
  { file: "下楼.csv", label: "下楼" },
  { file: "平地快跑.csv", label: "平地快跑" },
];
const colors = ["#2563eb", "#e11d48", "#16a34a", "#7c3aed"];

function parseCsv(text, file) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const expectedHeader = [
    "Time (s)",
    "Acceleration x (m/s^2)",
    "Acceleration y (m/s^2)",
    "Acceleration z (m/s^2)",
    "Absolute acceleration (m/s^2)",
  ];
  const header = lines[0].split(",");
  if (header.length !== expectedHeader.length || header.some((value, index) => value !== expectedHeader[index])) {
    throw new Error(`${file}: CSV 表头不符合预期`);
  }

  const rows = lines.slice(1).map((line, index) => {
    const values = line.split(",").map(Number);
    if (values.length !== expectedHeader.length || values.some(value => !Number.isFinite(value))) {
      throw new Error(`${file}: 第 ${index + 2} 行包含无效数据`);
    }
    return values;
  });
  if (rows.length < 2) {
    throw new Error(`${file}: 有效采样数据不足`);
  }
  for (let index = 1; index < rows.length; index += 1) {
    if (rows[index][0] <= rows[index - 1][0]) {
      throw new Error(`${file}: 时间戳必须严格递增`);
    }
  }
  return rows;
}

function quantile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * fraction)];
}

function populationStandardDeviation(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

function format(value) {
  return value.toFixed(3);
}

function linePath(rows, column, x, y, width, height, min, max, stride) {
  const range = max - min || 1;
  const points = [];
  for (let index = 0; index < rows.length; index += stride) {
    const px = x + (rows[index][0] - rows[0][0]) / (rows.at(-1)[0] - rows[0][0]) * width;
    const py = y + height - (rows[index][column] - min) / range * height;
    points.push(`${points.length === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`);
  }
  const last = rows.at(-1);
  if ((rows.length - 1) % stride !== 0) {
    const px = x + width;
    const py = y + height - (last[column] - min) / range * height;
    points.push(`L${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return points.join(" ");
}

function renderChart(label, rows) {
  const width = 1200;
  const height = 760;
  const left = 92;
  const plotWidth = 1040;
  const topY = 116;
  const topHeight = 300;
  const bottomY = 500;
  const bottomHeight = 150;
  const duration = rows.at(-1)[0] - rows[0][0];
  const stride = Math.max(1, Math.floor(rows.length / 1600));
  const axisValues = rows.flatMap(row => row.slice(1, 4));
  const axisMin = Math.floor(Math.min(...axisValues) - 0.5);
  const axisMax = Math.ceil(Math.max(...axisValues) + 0.5);
  const magnitudeValues = rows.map(row => row[4]);
  const magnitudeMin = Math.floor((Math.min(...magnitudeValues) - 0.5) * 2) / 2;
  const magnitudeMax = Math.ceil((Math.max(...magnitudeValues) + 0.5) * 2) / 2;
  const ticks = 4;
  const axisNames = ["X", "Y", "Z"];
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#fff"/>
<style>text{font-family:"Microsoft YaHei",Arial,sans-serif;fill:#1f2937}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.4}.label{font-size:15px}.small{font-size:13px}.title{font-size:27px;font-weight:700}.subtitle{font-size:16px;fill:#4b5563}</style>
<text x="${left}" y="46" class="title">${label}三轴加速度波形</text>
<text x="${left}" y="76" class="subtitle">采样 ${rows.length} 点 · ${format(1 / ((rows.at(-1)[0] - rows[0][0]) / (rows.length - 1)))} Hz · 时长 ${format(duration)} s</text>`;

  for (let tick = 0; tick <= ticks; tick += 1) {
    const fraction = tick / ticks;
    const y = topY + fraction * topHeight;
    const axisValue = axisMax - fraction * (axisMax - axisMin);
    const magnitudeValue = magnitudeMax - fraction * (magnitudeMax - magnitudeMin);
    svg += `<line x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}" class="grid"/>`;
    svg += `<text x="${left - 12}" y="${y + 5}" text-anchor="end" class="small">${format(axisValue)}</text>`;
    if (tick <= ticks) {
      const bottomGridY = bottomY + fraction * bottomHeight;
      svg += `<line x1="${left}" y1="${bottomGridY}" x2="${left + plotWidth}" y2="${bottomGridY}" class="grid"/>`;
      svg += `<text x="${left - 12}" y="${bottomGridY + 5}" text-anchor="end" class="small">${format(magnitudeValue)}</text>`;
    }
  }
  svg += `<line x1="${left}" y1="${topY}" x2="${left}" y2="${topY + topHeight}" class="axis"/>`;
  svg += `<line x1="${left}" y1="${topY + topHeight}" x2="${left + plotWidth}" y2="${topY + topHeight}" class="axis"/>`;
  svg += `<text x="26" y="${topY + topHeight / 2}" transform="rotate(-90 26 ${topY + topHeight / 2})" text-anchor="middle" class="label">加速度 (m/s²)</text>`;
  svg += `<text x="${left}" y="${topY - 12}" class="label">三轴分量</text>`;
  for (let axis = 0; axis < 3; axis += 1) {
    svg += `<path d="${linePath(rows, axis + 1, left, topY, plotWidth, topHeight, axisMin, axisMax, stride)}" fill="none" stroke="${colors[axis]}" stroke-width="1.4" opacity="0.9"/>`;
    svg += `<text x="${left + axis * 112}" y="${topY + topHeight + 34}" fill="${colors[axis]}" class="label">${axisNames[axis]} 轴</text>`;
  }

  const magnitudeY = bottomY + bottomHeight - (gravity - magnitudeMin) / (magnitudeMax - magnitudeMin) * bottomHeight;
  svg += `<line x1="${left}" y1="${bottomY}" x2="${left}" y2="${bottomY + bottomHeight}" class="axis"/>`;
  svg += `<line x1="${left}" y1="${bottomY + bottomHeight}" x2="${left + plotWidth}" y2="${bottomY + bottomHeight}" class="axis"/>`;
  svg += `<line x1="${left}" y1="${magnitudeY}" x2="${left + plotWidth}" y2="${magnitudeY}" stroke="#6b7280" stroke-dasharray="7 5"/>`;
  svg += `<path d="${linePath(rows, 4, left, bottomY, plotWidth, bottomHeight, magnitudeMin, magnitudeMax, stride)}" fill="none" stroke="#111827" stroke-width="1.5"/>`;
  svg += `<text x="${left}" y="${bottomY - 14}" class="label">绝对加速度模长 |a| (m/s²)，虚线为重力参考值 9.81 m/s²</text>`;
  svg += `<text x="26" y="${bottomY + bottomHeight / 2}" transform="rotate(-90 26 ${bottomY + bottomHeight / 2})" text-anchor="middle" class="label">|a| (m/s²)</text>`;
  for (let tick = 0; tick <= ticks; tick += 1) {
    const x = left + tick / ticks * plotWidth;
    svg += `<line x1="${x}" y1="${topY + topHeight}" x2="${x}" y2="${topY + topHeight + 5}" class="axis"/>`;
    svg += `<line x1="${x}" y1="${bottomY + bottomHeight}" x2="${x}" y2="${bottomY + bottomHeight + 5}" class="axis"/>`;
    svg += `<text x="${x}" y="${bottomY + bottomHeight + 26}" text-anchor="middle" class="small">${format(duration * tick / ticks)} s</text>`;
  }
  svg += `<text x="${left + plotWidth / 2}" y="${height - 35}" text-anchor="middle" class="label">相对时间 (s)</text></svg>`;
  return svg;
}

function summarize(label, rows) {
  const duration = rows.at(-1)[0] - rows[0][0];
  const sampleRate = (rows.length - 1) / duration;
  const windowSize = Math.round(sampleRate);
  const windowScores = [];
  for (let start = 0; start + windowSize <= rows.length; start += windowSize) {
    windowScores.push(populationStandardDeviation(rows.slice(start, start + windowSize).map(row => row[4])));
  }
  return {
    label,
    rows: rows.length,
    duration,
    sampleRate,
    magnitudeSd: populationStandardDeviation(rows.map(row => row[4])),
    windowP10: quantile(windowScores, 0.1),
    windowMedian: quantile(windowScores, 0.5),
    windowP90: quantile(windowScores, 0.9),
  };
}

await mkdir(chartDirectory, { recursive: true });
const summaries = [];
for (const sample of samples) {
  const csv = await readFile(path.join(outputDirectory, sample.file), "utf8");
  const rows = parseCsv(csv, sample.file);
  summaries.push(summarize(sample.label, rows));
  await writeFile(path.join(chartDirectory, `${sample.label}.svg`), renderChart(sample.label, rows), "utf8");
}

const summaryHeader = "状态,采样点数,时长_s,采样频率_Hz,全程绝对加速度标准差_m_s2,1秒窗口标准差P10_m_s2,1秒窗口标准差中位数_m_s2,1秒窗口标准差P90_m_s2";
const summaryRows = summaries.map(item => [
  item.label,
  item.rows,
  format(item.duration),
  format(item.sampleRate),
  format(item.magnitudeSd),
  format(item.windowP10),
  format(item.windowMedian),
  format(item.windowP90),
].join(","));
await writeFile(path.join(outputDirectory, "样本统计.csv"), `\uFEFF${summaryHeader}\n${summaryRows.join("\n")}\n`, "utf8");
console.log(`已生成 ${summaries.length} 张 SVG 波形图和样本统计.csv。`);
