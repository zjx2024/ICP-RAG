#!/bin/bash
# 停止知识图谱全部服务（配套 start.sh / restart-app.sh）
cd "$(dirname "$0")"

echo "=== 停止知识图谱服务 ==="
docker compose -p icp down

echo ""
echo "=== 服务已停止 ==="
