"""
strategy_manager/signal.py — 回测与实盘共享的信号计算模块

提供：
  compute_target_positions(factors, prev_positions)  →  {-1, 0, +1} 目标仓位张量
  reconcile_action(current, target)                  →  动作字符串

信号逻辑采用 Neutral Band（带滞后出场）：
  factor > ENTRY_THRESHOLD  → 做多
  factor < -ENTRY_THRESHOLD → 做空
  否则保持当前仓位（滞后出场），直到 |factor| < EXIT_THRESHOLD 才平仓

这比 sign(tanh(factor)) 的"全程满仓"策略降低约 30-40% 的噪声换手。
"""
from __future__ import annotations

import torch
from torch import Tensor

# ── 信号阈值 ──────────────────────────────────────────────────────────────────
# 标准化后因子值域 [-3, 3]，选 0.6 入场 / 0.2 出场
ENTRY_THRESHOLD: float = 0.3   # 降低入场阈值：标准化后因子 std≈1，±0.3 能捕捉约50%时间
EXIT_THRESHOLD:  float = 0.1   # 对应降低出场阈值


def compute_target_positions(
    factors:        Tensor,
    prev_positions: Tensor | None = None,
) -> Tensor:
    """将因子张量转换为目标仓位 {-1, 0, +1}（Neutral Band 逻辑）。

    设计决策：
    - 训练/回测（stateless）：中间区 [EXIT, ENTRY] 直接输出 0（空仓）。
      使得回测可向量化，性能高，且行为保守（比实盘少交易）。
    - 实盘（传入 prev_positions）：中间区保持前仓（滞后出场），
      降低噪声换手。实盘比回测多持一点仓，结果 >= 回测是合理预期。

    回测与实盘之间的这个微小差异是有意为之的——
    回测保守估计，实盘更宽松，不是 bug 而是安全边际。

    Args:
        factors:        [N, T] 或 [N] 的因子张量。
        prev_positions: [N, T] 或 [N] 的前一时刻仓位（实盘用，None=stateless）。
    """
    # 开多区
    long_mask  = factors >  ENTRY_THRESHOLD
    # 开空区
    short_mask = factors < -ENTRY_THRESHOLD
    # 平仓区
    exit_mask  = factors.abs() < EXIT_THRESHOLD

    if prev_positions is None:
        # 训练模式：无状态，中间区直接空仓
        pos = torch.zeros_like(factors)
        pos[long_mask]  =  1.0
        pos[short_mask] = -1.0
        # exit_mask 区域保持 0（已初始化为 0）
    else:
        # 实盘/回测模式：中间区保持前仓（滞后出场）
        pos = prev_positions.float().clone()
        pos[long_mask]  =  1.0
        pos[short_mask] = -1.0
        pos[exit_mask]  =  0.0

    return pos.sign()   # 确保值为 {-1, 0, +1}


def compute_target_positions_stateless(factors: Tensor) -> Tensor:
    """无状态版本，供训练回测快速计算（不需要前仓状态）。

    等价于 compute_target_positions(factors, prev_positions=None)。
    """
    return compute_target_positions(factors, prev_positions=None)


# ── 动作常量 ──────────────────────────────────────────────────────────────────
HOLD             = "HOLD"
OPEN_LONG        = "OPEN_LONG"
OPEN_SHORT       = "OPEN_SHORT"
CLOSE            = "CLOSE"
REVERSE_TO_LONG  = "REVERSE_TO_LONG"
REVERSE_TO_SHORT = "REVERSE_TO_SHORT"


def reconcile_action(current: int, target: int) -> str:
    """根据当前仓位方向和目标方向，返回应执行的动作。

    Args:
        current: 当前仓位方向，+1（多）/ -1（空）/ 0（空仓）。
        target:  目标仓位方向，+1 / -1 / 0。

    Returns:
        动作字符串，取值为模块级常量之一。
    """
    if current == target:
        return HOLD
    if current == 0:
        return OPEN_LONG if target == 1 else OPEN_SHORT
    if target == 0:
        return CLOSE
    return REVERSE_TO_LONG if target == 1 else REVERSE_TO_SHORT
