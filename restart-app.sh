#!/bin/bash
# 重建并重启 app 服务（改代码后的快速迭代，不影响 postgres/es/hugegraph 等）
# 配套 start.sh / stop.sh
cd "$(dirname "$0")"

echo "=== 重建应用镜像并重启 ==="
echo "停止应用服务..."
docker compose -p icp -f docker-compose.yml stop app

echo "重建应用镜像..."
docker compose -p icp -f docker-compose.yml build app

echo "启动应用服务..."
docker compose -p icp -f docker-compose.yml up -d app

echo ""
echo "=== 服务状态 ==="
docker compose -p icp -f docker-compose.yml ps