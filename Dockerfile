# ===== Frontend build =====
FROM node:20-alpine AS frontend-builder

WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# ===== Python dependency wheels =====
FROM python:3.10-slim AS python-builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ===== Python runtime =====
FROM python:3.10-slim

WORKDIR /app

# Compilers and headers stay in python-builder; the runtime keeps only the
# PostgreSQL shared library and the entrypoint's TCP readiness helper.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY --from=python-builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

COPY backend/ backend/
COPY scripts/ scripts/
COPY static/ static/
# 始终用当前源码构建出的 ICP 专用前端覆盖仓库内历史静态产物。
COPY --from=frontend-builder /src/static/dist/ static/dist/
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh \
    # 与 docker compose 的 ${APP_UID:-1000} 保持一致：容器默认非 root 运行，
    # 避免以 root 写入 volumes/artifacts 后宿主普通用户无法解析/回收。
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin icp \
    && mkdir -p /app/data/artifacts \
    && chown -R 1000:1000 /app

USER 1000:1000

EXPOSE 9999
ENTRYPOINT ["/app/entrypoint.sh"]
