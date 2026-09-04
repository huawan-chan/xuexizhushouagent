@echo off
chcp 65001 >nul
title 知友 AI · 一键启动
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python。
    echo   请先安装 Python 3.10 - 3.12: https://www.python.org/downloads/
    echo   安装时务必勾选 "Add Python to PATH"。
    pause
    exit /b 1
)

if not exist .venv (
    echo [1/4] 首次运行，创建虚拟环境...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo [提示] 已生成 .env 文件。请打开它，把 GROQ_API_KEY 改成你的密钥：
    echo       免费申请: https://console.groq.com/keys
    echo       改完后重新双击本脚本即可。
    pause
    exit /b 1
)

findstr /C:"your_groq_api_key_here" .env >nul
if not errorlevel 1 (
    echo [提示] .env 里还是示例占位符，请填入 GROQ_API_KEY 后重新运行。
    echo       免费申请: https://console.groq.com/keys
    pause
    exit /b 1
)

echo [2/4] 安装依赖(首次约几分钟，之后秒过)...
python -m pip install -r requirements.txt --disable-pip-version-check -q

echo [3/4] 启动服务...
echo.
echo   网页地址:  http://127.0.0.1:8001   浏览器打开即可对话
echo   关闭窗口或按 Ctrl+C 停止。
echo.
python -m web.fastapi_app

pause
