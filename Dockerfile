# 知友 AI · 单容器镜像(默认 FastAPI + SSE 界面, 端口 8001)
# 用途: 部署到 Hugging Face Spaces / Render / 任意支持 Docker 的平台，
#       部署一次后把网址发给别人即可使用，对方无需安装任何东西。
#
# 密钥通过运行时环境变量注入(不要写进镜像)：
#   GROQ_API_KEY=...           必填
#   GROQ_MODEL=...             可选
#   TAVILY_API_KEY=...         可选

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_UI=fastapi

WORKDIR /app

# 先装依赖，利用镜像层缓存(改代码不会重新装依赖)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码与前端页面
COPY . .

# 数据目录(知识库 PDF 放入 /app/data/pdfs；需持久化可挂载卷)
RUN mkdir -p /app/data/pdfs

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8001/api/health', timeout=3)"

CMD ["python", "-m", "web.fastapi_app"]
