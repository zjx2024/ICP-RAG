FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 只拷贝重构后的运行代码（不再 COPY . . 引入旧 app/ 代码、frontend 源码、docs 等）
# - backend/: 全部后端分层代码（唯一运行依赖）
# - static/: 前端构建产物 static/dist + 旧版单页回滚备份（SPA 由 backend.main 服务）
# - entrypoint.sh: 启动脚本
COPY backend/ backend/
COPY static/ static/
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

EXPOSE 9999

ENTRYPOINT ["/app/entrypoint.sh"]