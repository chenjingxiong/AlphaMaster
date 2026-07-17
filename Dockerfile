# syntax=docker/dockerfile:1.6
# AlphaMaster - 量化因子挖掘中心
# 多阶段构建：builder 层编译/安装依赖，runtime 层只拷贝最小运行集，最小化镜像体积。
#
# Web 控制台默认端口 8765（run_web.py）。实盘/run.py 需要 MT5 终端，仅在 Windows 可用，
# 容器内默认跑 Web + 训练 + 回测路径（CPU），MetaTrader5 / tvdatafeed 为可选依赖，
# 缺失时 config.py 与各 data_source 会自动降级，不影响 Web 启动。

# ─────────────────────────────────────────────────────────────
# Stage 1: builder —— 在虚拟环境里安装全部依赖（含编译产物），随后剥离以瘦身
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# torch CPU 版显著减小体积（去掉 CUDA）。先装 torch，再装其余依赖。
# build-essential / git：部分包（如 cryptography、tvdatafeed）需要编译或 git 拉取。
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境，便于 runtime 阶段整体拷贝且不污染系统解释器
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# 先拷贝依赖清单，利用层缓存：requirements 不变时跳过耗时的 pip install
# 使用容器专用清单（去掉只有 Windows wheel 的 MetaTrader5；torch 由下面单独装 CPU 版）
COPY requirements-docker.txt ./

# 两步安装，确保 torch 只装 CPU 版（~200MB vs CUDA ~2GB）：
#   1) --index-url 指向 PyTorch CPU 索引，仅安装 torch + numpy（torch 依赖）
#   2) 恢复 PyPI 默认源，安装其余依赖（tvdatafeed 来自 git，builder 已装 git）
RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch numpy \
    && pip install --no-cache-dir -r requirements-docker.txt


# ─────────────────────────────────────────────────────────────
# Stage 2: runtime —— 最小化运行镜像
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AlphaMaster" \
      org.opencontainers.image.source="https://github.com/chenjingxiong/AlphaMaster" \
      org.opencontainers.image.description="量化因子挖掘中心 Web 控制台 + 训练/回测引擎"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    TZ=Asia/Shanghai \
    PYTHONPATH=/app \
    # 覆盖 config.py 中默认的 Windows K 线缓存目录
    KLINE_CACHE_DIR=/app/data/kline_cache

# 仅装运行所需系统库（libgomp 供 torch OpenMP；curl 供 healthcheck；tzdata 时区）。
# 不装 git/build-essential，体积更小、攻击面更小。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 从 builder 拷贝已装好依赖的虚拟环境（已经是 CPU-only、无 __pycache__）
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# 拷贝应用源码。.dockerignore 已排除 data/checkpoints/strategies/*.log 等。
COPY . .

# 运行期需要的目录（compose 也会挂卷，这里保证镜像单独跑也成立）
RUN mkdir -p /app/data/kline_cache /app/strategies /app/checkpoints /app/backtest_output

EXPOSE 8765

# 健康检查：Web 控制台 /api/health 端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/api/health || exit 1

# 默认启动 Web 控制台，监听 0.0.0.0:8765（容器外可访问）。
# 如需训练/回测 CLI，可覆盖 command，例如：
#   docker run ... python train_file.py --data-file /app/data/BTCUSDT_H1.parquet
CMD ["python", "run_web.py", "--host", "0.0.0.0", "--port", "8765"]
