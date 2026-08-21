# ===== Frontend build =====
FROM node:20-alpine AS frontend-builder

WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# ===== Python runtime =====
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

COPY backend/ backend/
COPY static/ static/
# 始终用当前源码构建出的 ICP 专用前端覆盖仓库内历史静态产物。
COPY --from=frontend-builder /src/static/dist/ static/dist/
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

EXPOSE 9999
ENTRYPOINT ["/app/entrypoint.sh"]
