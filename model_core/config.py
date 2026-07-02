"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 训练参数（阶段 A：找简单稳定公式）────────────────────────────────
    # 阶段 A（当前）：MAX_FORMULA_LEN=8，TRAIN_STEPS=300
    #   目标：找简单、稳定、低换手的基础公式，防过拟合
    # 阶段 B（完成阶段A后切换）：
    #   MAX_FORMULA_LEN=14，TRAIN_STEPS=500，ELITE_REPLAY_FRAC=0.35
    #   目标：围绕好公式附近做组合增强
    BATCH_SIZE      = 128
    TRAIN_STEPS     = 300   # 每品种训练步数（多因子模式下每品种独立跑）
    MAX_FORMULA_LEN = 8     # 阶段B改为 14

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控而非线性加权 ────────────────────
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.002  # 0.005→0.002：降低门控阈值，让IC更频繁参与调节
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.85

    # ── 熵保护 ─────────────────────────────────────────────────────────
    ENTROPY_COEFF_MAX:   float = 0.50
    ENTROPY_COEFF_POWER: float = 1.3
    ENTROPY_COLLAPSE_THRESH: float = 0.5
    ENTROPY_COLLAPSE_STEPS:  int   = 15

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25   # 阶段A；阶段B改为 0.35
    ELITE_POOL_SIZE:    int   = 30
    ELITE_REWARD_SCALE: float = 1.2

    # ── 坍塌重启 ───────────────────────────────────────────────────────
    MAX_RESTARTS:   int   = 8
    RESTART_NOISE:  float = 0.05

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 10
    CORR_THRESHOLD:   float = 0.7
    CORR_PENALTY:     float = 0.5

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    WF_GAP: int = 20
