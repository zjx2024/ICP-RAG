#!/bin/bash
cd "$(dirname "$0")"

echo "启动 ICP_RAG 新一代知识图谱服务（热启动模式）..."
echo "监听端口: 19999"
echo "修改代码后会自动重载，无需手动重启"
echo ""
echo "前置：请先运行 ./start.sh 启动基础设施（postgres 5433 / ES 9201 / minio 9002 / hugegraph 8082）"
echo ""

# ===== 独立基础设施：宿主机端口映射（对应 docker-compose.yml，与 RAG_Agent_v1.0 完全隔离）=====
export ES_PORT=9201
export POSTGRES_PORT=5433
export MINIO_ENDPOINT=localhost:9002
export HUGEGRAPH_PORT=8082
# SERVER_PORT 同时被 docx_parser 用于拼接图片 URL，必须与下方 uvicorn 端口一致
export SERVER_PORT=19999

# 宿主机直跑时，图片目录需指向 docker-compose 的挂载目录（./app/storage/images），
# 与镜像模式保持同一份图片，避免画廊图片落到 backend/storage/images 导致不一致
export IMAGE_STORAGE_DIR="$(pwd)/app/storage/images"

# 只监听 backend 源码目录：避免 --reload 全目录扫描 volumes/（root 属主不可读）导致 reload 守护崩溃
/home/kdfy/.local/bin/uvicorn backend.main:app --host 0.0.0.0 --port 19999 --log-level info --reload --reload-dir backend