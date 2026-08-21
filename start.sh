#!/usr/bin/env bash
# 知识图谱服务启动脚本
# 会自动检测宿主机局域网 IP（取默认路由接口的 IP），写入 .env 供 docker compose 注入容器
# 这样 Dify 对接页面上显示的 API 地址就能在局域网内被其他主机访问
#
# 配套脚本职责：
#   ./start.sh        启动全部服务（镜像已存在则直接启动，缺失才自动构建）
#   ./stop.sh         停止全部服务（docker compose down）
#   ./restart-app.sh  只重建并重启 app 服务（改代码后的快速迭代）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_NAME="icp"
ENV_FILE="$SCRIPT_DIR/.env"

# ---------- 自动检测局域网 IP ----------
detect_lan_ip() {
    local lan_ip=""

    # 1) 优先取默认路由接口的 IPv4（最可靠：就是访问外网用的网卡）
    if command -v ip >/dev/null 2>&1; then
        local default_iface
        default_iface=$(ip route show default 2>/dev/null | awk '/default/ {print $5; exit}')
        if [ -n "$default_iface" ]; then
            lan_ip=$(ip -4 addr show "$default_iface" 2>/dev/null \
                     | grep -oP 'inet\s+\K[0-9.]+' | head -1)
        fi
    fi

    # 2) 兜底：hostname -I 的第一个非 docker/loopback 地址
    if [ -z "$lan_ip" ] && command -v hostname >/dev/null 2>&1; then
        for ip in $(hostname -I 2>/dev/null); do
            case "$ip" in
                127.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) continue ;;
                *) lan_ip="$ip"; break ;;
            esac
        done
    fi

    # 3) 再兜底：用 socket 方式
    if [ -z "$lan_ip" ] && command -v python3 >/dev/null 2>&1; then
        lan_ip=$(python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
    s.close()
except Exception:
    pass
" 2>/dev/null)
    fi

    [ -z "$lan_ip" ] && lan_ip="127.0.0.1"
    echo "$lan_ip"
}

LAN_IP=$(detect_lan_ip)
echo "[INFO] 检测到宿主机局域网 IP: $LAN_IP"

# ---------- 写入/更新 .env ----------
write_env() {
    if [ -f "$ENV_FILE" ]; then
        # 已存在则更新 SERVER_IP 行（去掉旧值，避免重复）
        grep -v '^SERVER_IP=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
        mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi
    echo "SERVER_IP=$LAN_IP" >> "$ENV_FILE"
}

write_env
echo "[INFO] 已写入 $ENV_FILE"

# ---------- 启动 docker compose ----------
# 无参数时默认只启动基础设施（postgres/es/minio/hugegraph/hubble）：
# 应用服务（app）不随基础设施启动 —— 开发用 ./start_server.sh 源码热启动（uvicorn 19999），
# 避免容器版 app 占用 19999 端口与热启动冲突。
# 有参数时透传给 compose，例如：
#   ./start.sh up -d app          单独启动 docker 版应用（需先有 npm run build 产物）
#   ./start.sh up -d --build      强制重建镜像后启动
#   ./start.sh ps                 查看服务状态
if [ "$#" -eq 0 ]; then
    set -- up -d postgres elasticsearch minio hugegraph hubble
fi
echo "[INFO] 启动 docker compose (项目名: $PROJECT_NAME) ..."
exec docker compose -p "$PROJECT_NAME" -f "$SCRIPT_DIR/docker-compose.yml" "$@"
