# 知友 AI · 单容器镜像(默认 FastAPI + SSE 界面)
# 用途: 部署到 Hugging Face Spaces / Render / 任意支持 Docker 的平台，
#       部署一次后把网址发给别人即可使用，对方无需安装任何东西。
#
# 密钥通过运行时环境变量注入(不要写进镜像)，Hugging Face Spaces 的
# Settings -> Variables and secrets 中配置；若用 OpenAI 兼容接口(智谱/DeepSeek)：
#   LLM_PROVIDER=openai
#   OPENAI_API_KEY=xxx
#   OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
#   OPENAI_MODEL=glm-4-flash-250414
# 若用 Groq:
#   GROQ_API_KEY=xxx
#
# 注意: Hugging Face Spaces 只转发 7860 端口，容器内服务通过 PORT 环境变量
#       自动监听(web/fastapi_app.py 已支持)，无需改 CMD。

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_UI=fastapi

WORKDIR /app

# 先装依赖，利用镜像层缓存(改代码不会重新装依赖)。
# sentence-transformers 会拉 torch，这里固定装 CPU 版，避免下载几 GB 的 CUDA 包导致构建超时
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

# 再拷代码与前端页面
COPY . .

# 数据目录(知识库 PDF 放入 /app/data/pdfs；需持久化可挂载卷)
RUN mkdir -p /app/data/pdfs

# HF Spaces 要求容器监听 7860(PORT 环境变量注入)；本平台若无 PORT 则回退 8001
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request as u; import os; u.urlopen('http://127.0.0.1:' + os.getenv('PORT','7860') + '/api/health', timeout=3)"

CMD ["python", "-m", "web.fastapi_app"]
