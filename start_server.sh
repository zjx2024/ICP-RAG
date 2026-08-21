#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "启动 ICP 故障诊断知识系统（后端热重载）..."
echo "监听端口: 19999"
echo "前置：请先运行 ./start.sh 启动 PostgreSQL / ES / MinIO / HugeGraph"
echo ""

export ES_PORT=9201
export POSTGRES_PORT=5433
export MINIO_ENDPOINT=localhost:9002
export HUGEGRAPH_PORT=8082
export SERVER_PORT=19999

# 图片目录保存 FTA、RBD 与系统架构原图/证据，不再用于独立画廊。
export IMAGE_STORAGE_DIR="$(pwd)/app/storage/images"
mkdir -p "$IMAGE_STORAGE_DIR"

# backend.main 直接服务 static/dist；本地启动前默认编译当前 React 源码。
# 前端单独使用 npm run dev 时可设置 ICP_SKIP_FRONTEND_BUILD=1 跳过。
if [[ "${ICP_SKIP_FRONTEND_BUILD:-0}" != "1" ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[INFO] 构建 ICP 专用前端..."
    (
      cd frontend
      [[ -d node_modules ]] || npm ci
      npm run build
    )
  else
    echo "[WARN] 未找到 npm，将使用 static/dist 中已有构建产物"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -m uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 19999 \
  --log-level info \
  --reload \
  --reload-dir backend
