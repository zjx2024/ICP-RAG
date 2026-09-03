#!/usr/bin/env bash
# 知识图谱系统迁移打包脚本
# 用法: ./export-images.sh [输出目录，默认 ./images]
#
# 将 5 个运行时镜像导出为 tar 文件，并提示拷贝 volumes/ 数据目录，
# 生成一个可整体拷贝到新机器的迁移包（Dify 式部署）：
#
#   知识图谱迁移包/
#   ├── images/                    # 所有镜像 tar
#   │   ├── knowledge-graph-app.tar.gz
#   │   ├── postgres.tar.gz
#   │   ├── opensearch.tar.gz
#   │   ├── hugegraph.tar.gz
#   │   └── hubble.tar.gz
#   ├── volumes/                   # pg/opensearch/artifacts/hugegraph 及保留的历史 MinIO/ES 卷
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

# 各运行时服务对应的镜像名（与 docker-compose.yml 对齐）
declare -A IMAGES=(
    [knowledge-graph-app]="knowledge-graph-app:latest"
    [postgres]="postgres:15"
    [opensearch]="opensearchproject/opensearch:3.6.0"
    [hugegraph]="hugegraph/hugegraph:1.7.0"
    [hubble]="hugegraph/hubble:1.7.0"
)

# MinIO 不再是运行时依赖。如旧卷尚未完成迁移，只允许导出调用方
# 明确指定的不可变旧镜像 ID/digest，避免 latest 打开旧卷并改写元数据。
if [[ "${INCLUDE_LEGACY_MINIO_IMAGE:-0}" == "1" ]]; then
    if [[ -z "${LEGACY_MINIO_IMAGE:-}" ]]; then
        echo "[ERROR] INCLUDE_LEGACY_MINIO_IMAGE=1 时必须设置 LEGACY_MINIO_IMAGE"
        echo "        可用: docker inspect --format '{{.Image}}' <旧MinIO容器>"
        exit 1
    fi
    if [[ "$LEGACY_MINIO_IMAGE" == *:latest ]]; then
        echo "[ERROR] LEGACY_MINIO_IMAGE 不允许使用 :latest；请传旧容器的镜像 ID 或 digest"
        exit 1
    fi
    IMAGES[legacy-minio]="$LEGACY_MINIO_IMAGE"
fi

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
echo "  - volumes/           数据目录（先 ./stop.sh 停止服务再拷，保证一致性）"
echo "  - docker-compose.yml .env start.sh stop.sh restart-app.sh entrypoint.sh"
echo "  - scripts/migrate_minio_to_localfs.py requirements-migration.txt（如仍需迁移历史 MinIO）"
echo "  - volumes/minio/     仅作历史迁移源，校验完成前不要删除"
echo ""
echo "新机器恢复步骤:"
echo "  1. gunzip -c images/*.tar.gz | docker load（逐个加载镜像）"
echo "  2. 解压迁移包到目标目录，确认 volumes/ 结构完整"
echo "  3. ./start.sh up -d    （自动检测 IP 并启动全部 Docker 服务）"
echo "  4. 浏览器访问 http://<新机器IP>:19999"
