# 知友 AI · 学生辅导 + 科普问答 Agent

一个能直接跑起来的 **LangGraph ReAct Agent**：既是会**苏格拉底式引导**的家教（理科分步讲题、LaTeX 公式），也是会讲 **What-Why-Wow** 的科普老师。支持本地 PDF 知识库检索、计算器、联网搜索、长期学习画像记忆与 Web / CLI 双界面。

---

## 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| Agent 编排 | **LangGraph** | ReAct 循环：思考 → 调工具 → 回答；SQLite Checkpointer 管短期记忆 |
| 大模型 | **Groq(免费)**，可切 OpenAI 兼容 | `llama-3.3-70b-versatile`(默认)；Groq 在部分地区受限时，一行 `.env` 配置即可切换 DeepSeek / 智谱GLM / Kimi 等 |
| 知识库 | **Chroma + HuggingFace Embedding** | 本地向量库，免费离线 Embedding(中文用 `bge-small-zh-v1.5`) |
| Web UI | **Chainlit**(主) + **FastAPI + SSE**(备) | 流式打字机效果；FastAPI 自带轻量页面和 API |
| CLI | 标准库 argparse + asyncio | 命令行为主入口 |
| 记忆 | **sqlite** | 短期=会话(checkpoints.db)；长期=用户画像(memory.db) |

---

## 目录结构

```
.
├── agent/                # Agent 核心
│   ├── graph.py          # LangGraph ReAct 图、意图路由、流式封装、SQLite checkpointer
│   ├── tools.py          # 安全计算器 / RAG 检索 / 网页搜索(可选)
│   └── prompts.py        # 教学法(禁给答案) + What-Why-Wow 科普 Prompt
├── rag/
│   ├── loader.py         # PDF 解析 + 切分(data/pdfs → chunks)
│   └── vector_store.py   # Chroma 持久化、Embedding、检索/建库
├── memory/
│   └── session.py        # sqlite 长期用户画像(年级/学科/水平/薄弱点/偏好)
├── web/
│   ├── chainlit_app.py   # Chainlit 聊天界面(启动入口)
│   └── fastapi_app.py    # FastAPI + SSE(备选 UI 与 API)
├── cli.py                # 命令行入口
├── config.py             # 环境变量集中配置
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── data/                 # 运行时自动生成(pdfs/ 手动放资料, chroma_db/, *.db)
```

---

## 快速开始

### 1. 准备

```bash
# 申请免费 Groq Key: https://console.groq.com/keys
cp .env.example .env          # Windows: copy .env.example .env
# 编辑 .env，填入 GROQ_API_KEY=...
```

### 2. 安装依赖(建议 Python 3.10–3.12)

```bash
pip install -r requirements.txt
```

> 已适配 2026 年新生态(langgraph 1.x / langchain 1.x / chainlit 2.x)。若本机是
> Python 3.14 且安装 sentence-transformers 报错，请换用 Python 3.11/3.12。
> 首次运行会自动下载 embedding 模型(约 100MB)。若只想先测聊天、不想要 RAG，
> 可直接跑第 4 步，检索工具被调用时会提示“知识库为空”，不影响对话。

### 3. (可选) 建 PDF 知识库

```bash
# 把你的 PDF(讲义/教材)放进 data/pdfs/，然后：
python cli.py build-index
```

### 4. 跑起来

```bash
# 命令行对话(流式)
python cli.py chat

# 一次性提问
python cli.py ask "为什么天空是蓝色的？"

# 网页版(推荐, 打字机流式) → http://localhost:8000
chainlit run web/chainlit_app.py

# 备选 FastAPI 版 → http://localhost:8001
python -m web.fastapi_app
```

### Docker

```bash
cp .env.example .env && docker compose up -d
# http://localhost:8000 (Chainlit) 与 http://localhost:8001 (FastAPI)
```

---

## 让别人也能用(对方不用装 Python / 申请 key)

### A. 部署到云端 → 发一个网址(最省事)

仓库已带自包含 `Dockerfile`，可部署到免费平台(Hugging Face Spaces / Render 等)。
以 **Hugging Face Spaces** 为例：

1. 把仓库推到 GitHub；
2. 打开 https://huggingface.co/new-space ，**SDK 选 Docker**、空间名随意，关联你的 GitHub 仓库；
3. Space 的 **Settings → Variables and secrets** 添加：`GROQ_API_KEY=你的key`；
4. 等它构建完成，把生成的 `https://你的空间名.hf.space` 发给别人——对方**点开即用**。

> 云端容器里没装你的 PDF，知识库检索会提示"为空"；如需上传资料，把它放进项目
> `data/pdfs` 再构建，或用持久化存储挂载 `/app/data`。

### B. 局域网分享(Docker 已装)

```bash
cp .env.example .env      # 填入 GROQ_API_KEY
docker compose up -d
```
查本机局域网 IP：Windows `ipconfig` / macOS `ipconfig getifaddr en0`，
把 `你的IP:8000` 发给同一 Wi-Fi/内网的人即可。

### C. 发给对方自己跑(Windows 一键脚本)

确保对方装有 Python 3.10–3.12(安装时勾选 **Add Python to PATH**)，
把整个项目文件夹发给对方(不含 `data/`、`.venv/`)，双击 **`start.bat`**：
自动建虚拟环境 → 装依赖 → 启动 → 打开 http://127.0.0.1:8001。
对方仍需自行申请一个免费 Groq Key 填入 `.env`。

---

## 功能与使用

- **辅导模式**：发作业/题目即可触发。Agent 遵守“禁止直接给答案”铁律——先反问确认卡点，
  一次推进一小步，答对给肯定并提升难度，卡住就退回基础提问。数学公式用 LaTeX
  (`$...$` / `$$...$$`)，含计算的步骤会先调用 `math_calculator` 再引用结果。
  想强制开启可在句首写 `辅导:`。
- **科普模式**：句首“为什么/科普/介绍一下”或无明显学科词的问题自动进入。输出固定按
  **What → Why → Wow** 三段式；不确定的事实会尝试 `web_search`。
- **工具**：
  - `math_calculator`：AST 白名单安全求值，绝不 `eval` 任意代码。
  - `knowledge_base_search`：检索 `data/pdfs` 建的本地知识库，回答标注来源页码。
  - `web_search`(可选)：默认 DuckDuckGo 免费抓取；配置 `TAVILY_API_KEY` 后自动换用 Tavily。
- **记忆**：
  - 短期：`data/checkpoints.db` 保存完整多轮上下文(LangGraph thread)。
  - 长期：`data/memory.db` 保存学生画像(年级/学科/水平/薄弱点/学习偏好)，每轮对话后增量更新。
  - 画像查看：CLI `python cli.py profile`；清会话 `python cli.py reset`。

### CLI 内置命令

```
/exit 退出    /profile 查看我的长期画像    /reset 重置本次会话(保留画像)
```

---

## 流式与 API(FastAPI)

- `POST /api/chat`：`{"message": "...", "session_id": "u1"}` → `{"answer": "...", "mode": ...}`
- `GET /api/chat/stream?message=...&session_id=u1`：SSE，事件格式：

```
data: {"type":"start","mode":"science"}
data: {"type":"token","text":"天空之所以是…"}
data: {"type":"tool_result","tool":"knowledge_base_search","content":"…"}
data: {"type":"done","answer":"完整回答"}
```

`session_id` 即用户主键，不同 id 的短期会话与长期画像互相隔离。

---

## 自定义

- 换模型：`.env` 里改 `GROQ_MODEL`（如 `llama-3.1-8b-instant`）。
- 关网页搜索：`WEB_SEARCH_ENABLED=false`（tool 仍注册，调用时返回“未开启”）。
- 关 LLM 画像归纳(省 token)：`PROFILE_LLM_ON=false`，将退回纯关键词规则。
- 调难易度 / 引导节奏：改 `agent/prompts.py` 的 `TUTOR_RULES`。

## 常见问题

- **Groq 控制台/官网返回 `{"error":{"message":"Forbidden"}}`**：Groq 对部分网络地区
  (含中国大陆)不提供服务，这是服务商限制，换浏览器/重试无效。改用 OpenAI 兼容服务商即可
  (国内直连、有免费档)，改 `.env` 顶部后重启：
  ```ini
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-你的key
  OPENAI_BASE_URL=https://api.deepseek.com/v1
  OPENAI_MODEL=deepseek-chat
  ```
  常用免费/低价选择：DeepSeek(便宜、支持工具调用)、智谱 GLM(glm-4-flash 有免费档)、
  SiliconFlow(注册送额度、Qwen 系列支持工具)。注意 Agent 需**支持工具调用**的模型。
- **`RuntimeError: 缺少 GROQ_API_KEY`**：还没建 `.env` 或 key 没填（同理若缺
  `OPENAI_API_KEY`，先确认 `.env` 里 `LLM_PROVIDER=openai` 且四项都填了）。
- **首次建库很慢 / 需要联网**：正在下载 embedding 模型，之后离线可用。
- **只装了 CPU 版 torch 仍慢**：Embedding 模型本身很小；也可换更小的
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。
- **DuckDuckGo 抓不到结果**：网络受限或反爬；建议配 `TAVILY_API_KEY`（免费额度）。

## License

MIT —— 本项目只保留代码与 `.env.example`，密钥均在 `.env`（已被 `.gitignore` 忽略），可放心 push。
