# YAMNet 批量 WAV 评估工具 · 使用说明

> 位置：`D:\SmartLife\yamnet_prototype\`
> 程序：`yamnet_batch_eval.py`
> 模型：`models\yamnet\`（本地官方 SavedModel，**离线运行，不联网**）
> 环境：现成的 `.venv`，**不需要安装任何新依赖**

---

## 1. 这个工具做什么

对指定文件夹里的每个 `.wav`：

1. 读取并报告源格式（采样率、声道数、位深、时长、峰值/DC/削波）
2. 按 YAMNet 要求转换成 **16 kHz / 单声道 / float32 / [-1, 1]**
3. 跑一次推理，输出 **Top-5 类别 + 分数 + 推理耗时**
4. 汇总成功/失败数量，可选导出 JSON / CSV

它**只做评测**：不改主项目、不写数据库、不做 VAD、不做实时流式。

---

## 2. 放入真实录音

在 `yamnet_prototype` 下建一个目录，把 wav 丢进去即可，结构随意：

```
D:\SmartLife\yamnet_prototype\recordings\
├─ 门铃\
│   ├─ 2026-11-15_0930.wav
│   └─ 2026-11-15_1412.wav
├─ 婴儿哭\
│   └─ cry_01.wav
└─ 烟雾报警\
    └─ alarm_01.wav
```

**支持的格式**（自带解析器，不依赖 soundfile/librosa）：

| 项 | 支持范围 |
|---|---|
| 容器 | RIFF/WAVE（`.wav`）|
| 编码 | PCM 8/16/24/32-bit、IEEE float 32/64-bit、WAVE_FORMAT_EXTENSIBLE |
| 采样率 | 任意（8k/16k/44.1k/48k… 自动转到 16 kHz）|
| 声道 | 单声道 / 立体声 / 多声道（按通道取平均降为单声道）|

**录音建议**
- 时长 ≥ **1 秒**（YAMNet 一个分析窗是 0.96 s，短于它没有意义）
- 单段尽量只含**一个目标声源**，不要长时间静音+偶尔一声
- 峰值别顶到 0 dBFS（削波会让频谱失真），-6 ~ -12 dBFS 比较理想
- 手机录音的 `.m4a` / `.mp3` **不支持**，请先转成 wav（任意采样率都行）

> 目录里现在有 4 个 `selftest_*.wav` / `test_audio_16k.wav`，是我生成的**合成**测试音（不是真实录音），只用来验证链路，可随时删除。

---

## 3. 运行

先确保模型在 `models\yamnet\`（`saved_model.pb` 在，就不用再下载）：

```powershell
cd D:\SmartLife\yamnet_prototype
.\.venv\Scripts\python.exe yamnet_batch_eval.py --dir recordings
```

常用变体：

```powershell
# 含子目录（推荐：按类别分文件夹时用）
.\.venv\Scripts\python.exe yamnet_batch_eval.py --dir recordings --recursive

# 同时导出结果，便于后续统计
.\.venv\Scripts\python.exe yamnet_batch_eval.py --dir recordings --recursive `
    --json reports\run.json --csv reports\run.csv

# 只看前 3 名
.\.venv\Scripts\python.exe yamnet_batch_eval.py --dir recordings --top-k 3

# 指定其它模型目录（一般不用改）
.\.venv\Scripts\python.exe yamnet_batch_eval.py --dir recordings --model-dir models\yamnet
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--dir` | wav 所在文件夹 | `samples` |
| `--recursive` | 递归子目录 | 关 |
| `--top-k` | 输出前 K 名 | 5 |
| `--model-dir` | 本地模型目录 | `models\yamnet` |
| `--json` | 导出完整结果（含预处理统计）| 不导出 |
| `--csv` | 导出扁平表（每行一个类别）| 不导出 |

退出码：`0` 至少一个成功；`1` 全部失败；`2` 目录不存在/没有 wav。

---

## 4. 输出怎么读

```
[1/4] doorbell_01.wav
  source : 48000 Hz, 2 ch, 16-bit PCM, 2.50 s
  prep   : 16 kHz mono float32, peak 0.404 (-7.9 dBFS), dc -0.000
  info   : downmixed 2 ch -> 1 ch (channel mean)
  info   : resampled 48000 Hz -> 16000 Hz
  infer  : 5 frames, 15.7 ms (CPU)
  top-5 :
    1. Sine wave                              0.8031  ###################
    2. Beep, bleep                            0.1501  ####
    ...
```

| 字段 | 含义 |
|---|---|
| `source` | 原始文件真实格式（不是转换后的）|
| `prep` 峰值 / dBFS / dc | 转换后波形的峰值、峰值电平、直流偏置 |
| `info` / `warn` | 做过的处理（降混、重采样、裁剪）或风险提示（太短、太安静、削波、非有限值）|
| `frames` | 模型切出的帧数（0.96 s 窗、0.48 s 跳）|
| `infer` | 单次推理耗时（纯 CPU）|
| `top-5` | 按**帧间均值**聚合的 Top-5 类别与分数 |

---

## 5. 关于分数：请勿当成准确率或置信度

- 输出是 **521 个类别各自独立的 sigmoid 分数**，不是 softmax，**逐帧加起来不等于 1**（实测 1.3~1.8）
- 它是**模型对该类别的响应强度**，不等于“这段音频有 80% 概率是门铃”
- 模型只认识 AudioSet 的 521 个类，**你关心的声音可能根本没有对应类**（例如“微波炉叮”附近可能落在 `Beep, bleep` / `Ding`）
- 因此：**阈值必须用你自己的录音集标定**，不能直接套用别处的 0.5
- 正确说法是“模型分数 0.80”，不是“准确率 80%”

---

## 6. 预处理细节（便于排查异常）

1. **解码**：按 RIFF chunk 解析，支持 PCM/float/EXTENSIBLE；24-bit 手动拼字节，float 直接读
2. **声道**：多声道按**通道平均**降为单声道（不是取第一声道）
3. **数值**：非有限值（NaN/Inf）置 0 并告警；超出 [-1,1] 的样本裁剪并告警
4. **重采样**：FFT 频谱法（`rfft` → 截断或补零 → `irfft`），降采样时天然抗混叠
5. **不做**：不做归一化增益、不做降噪、不做 VAD 切分、不做分块拼接

---

## 7. 已知限制

- **FFT 重采样**在极短音频边缘会有轻微振铃；但目标只是 16 kHz 特征提取，对分类影响很小。若日后要与别的工具严格对齐，可再引入 `soxr`（需你批准装依赖）
- 整段一次推理，**超长音频**（如 10 分钟）会一次占用较多内存；建议按 3~10 秒切片后再测
- 纯 CPU，单文件约 15~200 ms（取决于时长），无 GPU 也可用
- 只支持 `.wav`；其它格式请先转换

---

## 8. 常见问题

**Q：文件没被扫到？**
只会匹配扩展名为 `.wav` 的文件；子目录需加 `--recursive`。

**Q：报 `duration ... too short`？**
短于 0.96 s 会给不出有意义的帧，建议 ≥1 s。

**Q：报 `very quiet` 或 `signal is digital silence`？**
录音增益太低或根本没录到；这些文件的分数没有参考价值。

**Q：立体声/48 kHz 会影响结果吗？**
不会，工具内部已统一转换；`source` 行显示的是原始格式，便于你核对。

**Q：结果看着不对？**
先确认这段音频里到底是哪 521 类之一（对照 `models\yamnet\assets\yamnet_class_map.csv`），再判断阈值。真实场景还建议先做 VAD 切出有声音的片段。

---

## 9. 文件清单

| 文件 | 作用 |
|---|---|
| `yamnet_batch_eval.py` | **本批工具**（批量评测）|
| `yamnet_smoke_test.py` | 冒烟测试（未改动，保留原样）|
| `download_yamnet.py` | 模型下载/校验/解压（可复现）|
| `models\yamnet\` | 本地 SavedModel + `assets\yamnet_class_map.csv` |
| `samples\` | 合成测试音（可删）|
| `reports\demo_result.json` | 本工具的一次示例输出 |
