#!/usr/bin/env bash
# 知识图谱系统迁移打包脚本
# 用法: ./export-images.sh [输出目录，默认 ./images]
#
# 将 6 个服务镜像导出为 tar 文件 + 拷贝 volumes/ 数据目录，
# 生成一个可整体拷贝到新机器的迁移包（Dify 式部署）：
#
#   知识图谱迁移包/
#   ├── images/                    # 所有镜像 tar
#   │   ├── knowledge-graph-app.tar.gz
#   │   ├── postgres.tar.gz
#   │   ├── elasticsearch.tar.gz
#   │   ├── minio.tar.gz
#   │   ├── hugegraph.tar.gz
#   │   └── hubble.tar.gz
#   ├── volumes/                   # 全部持久化数据（pg/es/minio/hugegraph）
#   ├── docker-compose.yml         # 编排配置
#   ├── .env                       # 环境变量（含密码，注意安全）
#   ├── start.sh / stop.sh / restart-app.sh
#   └── entrypoint.sh
#
# 新机器恢复步骤见 docs/部署指南.md

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="${1:-images}"
mkdir -p "$OUT_DIR"

# 各服务对应的镜像名（与 docker-compose.yml 对齐）
declare -A IMAGES=(
    [knowledge-graph-app]="knowledge-graph-app:latest"
    [postgres]="postgres:15"
    [elasticsearch]="elasticsearch:8.14.0"
    [minio]="minio/minio:latest"
    [hugegraph]="hugegraph/hugegraph:latest"
    [hubble]="hugegraph/hubble:latest"
)

echo "[INFO] 导出镜像到 $OUT_DIR/ ..."
for name in "${!IMAGES[@]}"; do
    img="${IMAGES[$name]}"
    out="$OUT_DIR/${name}.tar.gz"
    if docker image inspect "$img" >/dev/null 2>&1; then
        echo "  - $img -> $out"
        docker save "$img" | gzip > "$out"
    else
        echo "  [SKIP] 镜像不存在: $img"
    fi
done

echo "[INFO] 镜像导出完成"
echo ""
echo "[INFO] 请手动拷贝以下内容到新机器（迁移包）:"
echo "  - $OUT_DIR/          全部镜像 tar"
echo "  - volumes/           数据目录（可先 ./stop.sh 停止服务再拷，保证一致性）"
echo "  - docker-compose.yml .env start.sh stop.sh restart-app.sh entrypoint.sh"
echo ""
echo "新机器恢复步骤:"
echo "  1. gunzip -c images/*.tar.gz | docker load（逐个加载镜像）"
echo "  2. 解压迁移包到目标目录，确认 volumes/ 结构完整"
echo "  3. ./start.sh          （自动检测 IP 并启动全部服务）"
echo "  4. 浏览器访问 http://<新机器IP>:9999"
