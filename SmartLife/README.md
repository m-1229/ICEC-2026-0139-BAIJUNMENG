# SmartLife

SmartLife 是面向听障人士的智能生活声音提醒辅助系统。本项目当前提供手机端网页、FastAPI 接口、SQLite 事件记录和提醒设置。

> **重要说明：** 当前没有麦克风监听，也没有接入 YAMNet 或其他 AI 模型。事件页面中的记录由用户手动模拟，数据库来源字段会标记为 `simulated`，不能当作真实声音识别结果。

## 项目结构

```text
SmartLife/
├── app/
│   ├── __init__.py
│   ├── database.py          # SQLite 初始化和数据访问
│   ├── main.py              # FastAPI 应用和 API 路由
│   └── schemas.py           # 请求、事件和设置的数据结构
├── data/                    # 首次启动时创建，保存本机数据库
├── static/
│   ├── css/styles.css       # 手机端页面样式
│   ├── js/app.js            # 前端交互与 API 请求
│   └── index.html           # 手机端首页
├── .gitignore
├── README.md
└── requirements.txt
```

数据库使用 Python 自带的 `sqlite3`，首次启动时自动创建 `data/smartlife.db` 和默认提醒设置。门铃、敲门、水壶和警报默认开启；更改设置会保存到本机数据库。不要手动删除数据库文件，否则本机的历史记录和设置也会丢失。

## 环境和启动（Windows PowerShell）

**运行识别栈必须使用 `yamnet_prototype\.venv`**：项目根目录的 `.venv` 只装了 FastAPI/uvicorn，
没有 numpy、sounddevice、tensorflow，用它启动会得到
`/api/listen/devices → available=false（No module named 'sounddevice'）`。

正式运行（启用敲门事件落库）：

```powershell
Set-Location D:\SmartLife
$env:SMARTLIFE_ENABLE_KNOCK_PERSIST = "1"
.\yamnet_prototype\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

要点：

- **环境变量必须在启动进程之前设置。** `SMARTLIFE_ENABLE_KNOCK_PERSIST` 在
  `app/recognition.py` 导入时读取一次，进程启动后再设置不会生效，必须重启进程。
  不设置或设为 `0` 时，敲门事件不写库（页面会显示“敲门事件落库未启用”）。
- **正式运行建议不要加 `--reload`。** reload 会派生工作子进程并反复重载模型，启动更慢、
  状态更不稳定，只适合开发调试。
- 需要在没有识别依赖的环境里只调接口时，才用根目录 `.venv`；此时不要打开麦克风监听。

单独验证音频依赖是否齐备：

```powershell
.\yamnet_prototype\.venv\Scripts\python.exe -c "import numpy, sounddevice, tensorflow, tensorflow_hub; print('ok')"
```

如果 `yamnet_prototype\.venv` 还没建好，先准备 Python 3.11 环境，再安装识别依赖
（tensorflow 2.15.1 / tensorflow-intel 2.15.1 / tensorflow-hub 0.16.1 / tf-keras 2.15.1 /
numpy 1.26.4 / sounddevice 0.5.6）。根目录 `.venv` 仍可按下面的方式创建，但只能用于无识别的接口调试：

```powershell
py -3.11 -m venv .venv
```

仅在虚拟环境里尚未安装 `requirements.txt` 时，才需要安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

SQLite 驱动是 Python 标准库的一部分，不需要额外安装。服务运行后打开 <http://127.0.0.1:8765/>；交互式 API 文档位于 <http://127.0.0.1:8765/docs>。

## API

- `GET /api/health`：服务健康检查和功能能力标记。
- `POST /api/test-alert`：不写入数据库的手动提醒通信测试。
- `POST /api/events/simulate`：创建并保存一条模拟事件。JSON 示例：

  ```json
  {
    "event_type": "doorbell",
    "confidence": 0.9
  }
  ```

  可用 `event_type`：`doorbell`、`knock`、`kettle`、`alarm`。置信度必须在 0 到 1 之间；事件来源由服务端设置为 `simulated`。

- `GET /api/events?limit=50&offset=0`：按发生时间倒序读取历史事件。`limit` 最大为 100。
- `GET /api/settings/alerts`：读取四类提醒开关。
- `PATCH /api/settings/alerts/{event_type}`：更新一个提醒开关。JSON 示例：

  ```json
  {
    "enabled": false
  }
  ```

无效事件类型、置信度和分页参数由 API 校验并返回错误；数据库查询使用参数化 SQL。

## 当前范围

已实现模拟事件记录、SQLite 历史查询、提醒设置持久化和网页展示。尚未实现真实声音采集/识别、误报过滤、用户测试记录、真实事件提醒/震动和 PWA 安装能力。当前服务仅监听本机地址，不支持手机通过局域网访问。
