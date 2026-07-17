#!/usr/bin/env bash
# scripts/deploy_remote.sh
# 在【能直连服务器的机器】（如本开发容器）上运行：
#   - SSH 到部署服务器
#   - （私有镜像需要时）登录 GHCR
#   - docker compose pull + 重启
#
# 用法：
#   scripts/deploy_remote.sh                          # 交互式输入服务器密码
#   DEPLOY_HOST=192.168.9.12 DEPLOY_USER=root \
#     SSHPASS=<password> scripts/deploy_remote.sh     # 非交互（CI/脚本）
#
# 前置条件：服务器上 /opt/alphamaster/ 下已有 docker-compose.yml（含 GHCR 镜像配置）。
# 首次部署：脚本会用 scp 把本仓库的 docker-compose.yml 上传到服务器并 docker compose up。
set -euo pipefail

DEPLOY_HOST="${DEPLOY_HOST:-192.168.9.12}"
DEPLOY_USER="${DEPLOY_USER:-root}"
DEPLOY_PORT="${DEPLOY_PORT:-22}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/alphamaster}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> 目标: ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_DIR}"

# 优先使用 sshpass（非交互），否则要求 ssh key 已配好
SSH_BASE=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p "${DEPLOY_PORT}")
if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "❌ SSHPASS 已设置但未安装 sshpass。安装: apt-get install -y sshpass" >&2
    exit 1
  fi
  SSH_BASE=(sshpass -p "$SSHPASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p "${DEPLOY_PORT}")
  SCP_BASE=(sshpass -p "$SSHPASS" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${DEPLOY_PORT}")
else
  SCP_BASE=(scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${DEPLOY_PORT}")
fi

# 1) 确保服务器上 compose 文件存在；不存在则从本仓库上传（首次部署）
REMOTE_HAS_COMPOSE=$("${SSH_BASE[@]}" "${DEPLOY_USER}@${DEPLOY_HOST}" \
  "test -f ${DEPLOY_DIR}/docker-compose.yml && echo yes || echo no" 2>/dev/null || echo "no")

"${SSH_BASE[@]}" "${DEPLOY_USER}@${DEPLOY_HOST}" "mkdir -p ${DEPLOY_DIR}"

if [[ "$REMOTE_HAS_COMPOSE" != "yes" ]]; then
  echo "==> 首次部署：上传 docker-compose.yml 到 ${DEPLOY_DIR}"
  "${SCP_BASE[@]}" "${REPO_ROOT}/docker-compose.yml" "${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_DIR}/docker-compose.yml"
  # 若本地有 .env，一并上传（含凭证，仅走 SSH 加密通道）
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    "${SCP_BASE[@]}" "${REPO_ROOT}/.env" "${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_DIR}/.env"
  fi
fi

# 2) GHCR 登录（私有镜像需要；GHCR_PULL_TOKEN 未设置则跳过——公开镜像无需登录）
GHCR_TOKEN="${GHCR_PULL_TOKEN:-${GHCR_TOKEN:-}}"
LOGIN_CMD=""
if [[ -n "$GHCR_TOKEN" ]]; then
  LOGIN_CMD="echo '${GHCR_TOKEN}' | docker login ghcr.io -u '${GHCR_USER:-chenjingxiong}' --password-stdin && "
fi

# 3) 拉取最新镜像 + 重启
echo "==> 拉取镜像并重启容器..."
"${SSH_BASE[@]}" "${DEPLOY_USER}@${DEPLOY_HOST}" "set -euo pipefail; \
  cd ${DEPLOY_DIR}; \
  ${LOGIN_CMD}\
  docker compose pull alphamaster 2>/dev/null || docker pull ghcr.io/chenjingxiong/alphamaster:latest; \
  docker compose up -d --force-recreate alphamaster 2>/dev/null || { \
    echo 'docker compose 不可用，尝试 docker run'; \
    docker rm -f alphamaster 2>/dev/null || true; \
    docker run -d --name alphamaster --restart unless-stopped \
      -p 8765:8765 \
      -v alphamaster_strategies:/app/strategies \
      -v alphamaster_checkpoints:/app/checkpoints \
      -v alphamaster_data:/app/data \
      -v alphamaster_backtest_output:/app/backtest_output \
      ghcr.io/chenjingxiong/alphamaster:latest; \
  }; \
  docker image prune -f >/dev/null; \
  echo '==> 等待健康检查...'; \
  for i in \$(seq 1 30); do \
    if curl -fsS http://127.0.0.1:8765/api/health >/dev/null 2>&1; then \
      echo '✅ AlphaMaster healthy on :8765'; \
      docker compose ps 2>/dev/null || docker ps --filter name=alphamaster; \
      exit 0; \
    fi; \
    sleep 2; \
  done; \
  echo '❌ Health check failed — 最近日志:'; \
  docker compose logs --tail=80 alphamaster 2>/dev/null || docker logs --tail=80 alphamaster; \
  exit 1"

echo "==> 部署完成：http://${DEPLOY_HOST}:8765"
