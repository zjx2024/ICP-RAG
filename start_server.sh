#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "启动 ICP 故障诊断知识系统（后端热重载）..."
echo "监听端口: 19999"
echo "前置：请先运行 ./start.sh 启动 PostgreSQL / OpenSearch / HugeGraph"
echo ""

# 本地热启动直接读取与 Compose 相同的连接配置，避免每次手工重复导出
# PostgreSQL 密码、模型地址等变量。仓库中的 .env 只允许使用 shell 兼容写法。
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${POSTGRES_PORT:-}" ]] && command -v docker >/dev/null 2>&1; then
  POSTGRES_PORT="$(docker compose -p icp port postgres 5432 2>/dev/null | tail -1 | awk -F: '{print $NF}')"
fi
# 普通用户无权访问 Docker socket 时，仍可从已解析的 Compose 配置取得宿主机端口；
# `docker compose config` 不访问 daemon，适合本地源码启动。
if [[ -z "${POSTGRES_PORT:-}" ]] && command -v docker >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  POSTGRES_PORT="$(
    docker compose -p icp config --format json 2>/dev/null \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["services"]["postgres"]["ports"][0]["published"])' \
      2>/dev/null || true
  )"
fi
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export OPENSEARCH_HOST="${OPENSEARCH_HOST:-localhost}"
export OPENSEARCH_PORT="${OPENSEARCH_PORT:-9201}"
export OPENSEARCH_SCHEME="${OPENSEARCH_SCHEME:-http}"
export OPENSEARCH_VERIFY_CERTS="${OPENSEARCH_VERIFY_CERTS:-false}"
export OPENSEARCH_INDEX_ALIAS="${OPENSEARCH_INDEX_ALIAS:-knowledge_graph_default}"
export OPENSEARCH_INDEX_VERSION="${OPENSEARCH_INDEX_VERSION:-v1}"
export STRICT_DEPENDENCIES="${STRICT_DEPENDENCIES:-true}"
export POSTGRES_PORT="${POSTGRES_PORT:-5433}"
export HUGEGRAPH_HOST="${HUGEGRAPH_HOST:-localhost}"
export HUGEGRAPH_PORT="${HUGEGRAPH_PORT:-8082}"
export SERVER_PORT="${SERVER_PORT:-19999}"

# 开发机可能设置了 HTTP(S) 代理。数据库旁路不受影响，但 requests 若把本机
# HugeGraph/Ollama 请求交给代理会得到 502。把本地依赖主机明确加入直连名单。
LOCAL_NO_PROXY="localhost,127.0.0.1,::1,${POSTGRES_HOST},${OPENSEARCH_HOST},${HUGEGRAPH_HOST}"
if [[ -n "${SERVER_IP:-}" ]]; then
  LOCAL_NO_PROXY+=",${SERVER_IP}"
fi
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

# 原件、解析产物和工程图证据统一使用本地 ArtifactStore。
# /images 只会服务 images 子目录，不会暴露其他原始文档。
export ARTIFACT_BACKEND=local
export ARTIFACT_STORAGE_DIR="$(pwd)/volumes/artifacts"
export ARTIFACT_TRASH_ENABLED=true
export IMAGE_STORAGE_DIR="$ARTIFACT_STORAGE_DIR/images"
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
