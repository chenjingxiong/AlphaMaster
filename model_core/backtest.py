"""
model_core/backtest.py — MT5 回测评估器（组合级多目标 Reward）

评分框架（5品种组合版）：
  final_score =
      0.35 * portfolio_sortino          # 组合整体风险调整收益
    + 0.20 * portfolio_calmar           # 组合整体回撤控制
    + 0.15 * ts_ic_stability            # 时序IC稳定性（比横截面IC更重要）
    + 0.10 * symbol_consistency         # 品种一致性（防止单品种拖累）
    + 0.10 * cost_stress                # 成本压力测试（2x成本下仍盈利）
    + 0.10 * turnover_quality           # 换手率质量（交易频率奖励）
    - complexity_penalty                # 公式长度惩罚
    - correlation_penalty               # 因子相关性惩罚（由 engine 施加）

symbol_consistency 规则：
  - N 个品种中至少 ceil(N*0.6) 个 Sortino > 0 → 正分
  - 任何品种 Sortino < -2.0 → 重惩罚
  - 全部品种 Sortino > 0 → 额外奖励
"""
import math
import torch
from torch import Tensor

from strategy_manager.signal import compute_target_positions_stateless

_H1_PERIODS_PER_YEAR = 6240
_SORTINO_CLIP        = 20.0


class MT5Backtest:
    """MT5 组合级回测评估器。"""

    def __init__(
        self,
        cost_rate:        float = 0.0001,
        periods_per_year: int   = _H1_PERIODS_PER_YEAR,
    ):
        self.cost_rate        = cost_rate
        self.periods_per_year = periods_per_year

    # ──────────────────────────────────────────────────────────────────────
    # 基础统计
    # ──────────────────────────────────────────────────────────────────────

    def _sortino(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        flat     = pnl.reshape(-1)
        mean_pnl = flat.mean()
        downside = flat[flat < 0]
        raw_std  = downside.std(unbiased=False) if downside.numel() > 0 \
                   else torch.tensor(0.0, dtype=flat.dtype, device=flat.device)
        floor          = torch.clamp(mean_pnl.abs(), min=eps)
        downside_std   = torch.clamp(raw_std, min=floor)
        sortino        = mean_pnl / downside_std * math.sqrt(self.periods_per_year)
        return torch.clamp(sortino, -_SORTINO_CLIP, _SORTINO_CLIP)

    def _calmar(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        """Calmar = annualized_return / max_drawdown（截断到 [-10, 10]）。"""
        flat      = pnl.reshape(-1)
        ann_ret   = flat.mean() * self.periods_per_year
        cum       = torch.cumsum(flat, dim=0)
        peak      = torch.cummax(cum, dim=0).values
        drawdown  = (peak - cum).max()
        drawdown  = torch.clamp(drawdown, min=eps)
        calmar    = ann_ret / drawdown
        return torch.clamp(calmar, -10.0, 10.0)

    # ──────────────────────────────────────────────────────────────────────
    # 组合级评分组件
    # ──────────────────────────────────────────────────────────────────────

    def _ts_ic_stability(self, factors: Tensor, target_ret: Tensor) -> float:
        """时序 IC 稳定性：每个品种内部 factor[t] 与 ret[t+1] 的相关性均值。

        比横截面 IC 更适合 5 品种宇宙（横截面 N=5 统计意义弱）。

        Returns:
            float，约 [-1, 1]，正值代表因子有预测力。
        """
        N, T = factors.shape
        if T < 10:
            return 0.0

        ic_list = []
        for n in range(N):
            x = factors[n, :-1]
            y = target_ret[n, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic.item())

        if not ic_list:
            return 0.0

        ic_mean = sum(ic_list) / len(ic_list)
        ic_std  = (sum((v - ic_mean) ** 2 for v in ic_list) / len(ic_list)) ** 0.5
        # 稳定性 = IC均值 / IC标准差（IR，截断到 [-3, 3]）
        stability = ic_mean / (ic_std + 1e-6)
        return float(max(-3.0, min(3.0, stability)))

    def _symbol_consistency(
        self,
        per_symbol_sortino: list[float],
        per_symbol_trade_count: list[int] | None = None,
    ) -> float:
        """品种一致性惩罚/奖励。

        规则（优先级从高到低）：
        1. 无交易品种超过 40%：重惩罚 -3.0（公式对多数品种无效）
        2. 任何品种 Sortino < -2.0：重惩罚 -2.0
        3. 有效品种（有交易）中正收益比例决定奖惩：
           - 有效正收益 < 60%：线性惩罚 [-1.0, 0)
           - 有效正收益 ≥ 60%：线性奖励 [0, 1.0]
           - 全部有效品种正收益：额外 +0.5

        Args:
            per_symbol_sortino:     每品种的 Sortino 值列表
            per_symbol_trade_count: 每品种的交易笔数（None 时不检查活跃度）
        """
        N = len(per_symbol_sortino)
        if N == 0:
            return 0.0

        # 1. 无交易品种超过 40%：最严重惩罚
        if per_symbol_trade_count is not None:
            n_inactive = sum(1 for c in per_symbol_trade_count if c == 0)
            inactive_ratio = n_inactive / N
            if inactive_ratio > 0.4:
                return -3.0

        # 2. 任何品种严重亏损：重惩罚
        if any(s < -2.0 for s in per_symbol_sortino):
            return -2.0

        # 3. 有效品种（有交易）中的正收益比例
        if per_symbol_trade_count is not None:
            # 只统计有交易的品种
            active_sortinos = [
                s for s, c in zip(per_symbol_sortino, per_symbol_trade_count) if c > 0
            ]
        else:
            active_sortinos = per_symbol_sortino

        if not active_sortinos:
            return -3.0   # 全部无交易

        n_positive = sum(1 for s in active_sortinos if s > 0)
        ratio = n_positive / len(active_sortinos)

        if ratio < 0.6:
            score = (ratio - 0.6) / 0.6 * 1.0   # [-1.0, 0)
        else:
            score = (ratio - 0.6) / 0.4 * 1.0   # [0, 1.0]

        if ratio == 1.0:
            score += 0.5   # 全部活跃品种正收益额外奖励

        return float(score)

    def _cost_stress(
        self,
        position:   Tensor,
        target_ret: Tensor,
        stress_mult: float = 2.0,
    ) -> float:
        """成本压力测试：2 倍成本下的 Sortino 是否还 > 0。

        Returns:
            float，压力测试 Sortino（截断到 [-5, 5]）。
        """
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        stressed_pnl = position * target_ret - turnover * self.cost_rate * stress_mult
        sortino = self._sortino(stressed_pnl)
        return float(torch.clamp(sortino, -5.0, 5.0))

    def _turnover_quality(self, position: Tensor) -> float:
        """交易频率质量奖励（每天约 1 笔为最优）。

        目标：每 12 bar 一笔（H1 每天约一笔）。
        """
        N, T = position.shape
        pos_2d = position.tolist()
        all_runs, total_trades = [], 0

        for n in range(N):
            runs, cur_len, cur_dir = [], 0, 0
            for p in pos_2d[n]:
                pi = int(p)
                if pi != 0:
                    if pi == cur_dir:
                        cur_len += 1
                    else:
                        if cur_len > 0: runs.append(cur_len)
                        cur_dir, cur_len = pi, 1
                else:
                    if cur_len > 0: runs.append(cur_len)
                    cur_dir, cur_len = 0, 0
            if cur_len > 0: runs.append(cur_len)
            all_runs.extend(runs)
            total_trades += len(runs)

        total_bars    = N * T
        target_trades = total_bars / 12.0
        actual_ratio  = total_trades / max(target_trades, 1.0)

        if actual_ratio <= 0:
            freq_score = -2.0
        elif actual_ratio < 0.05:
            freq_score = -2.0 + actual_ratio / 0.05
        elif actual_ratio < 0.5:
            freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
        elif actual_ratio <= 2.0:
            log_r = math.log(actual_ratio) / math.log(2.0)
            freq_score = 1.0 * math.exp(-0.5 * log_r ** 2)
        elif actual_ratio <= 8.0:
            freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
        else:
            freq_score = -2.0

        hold_bonus = 0.0
        if all_runs:
            avg_hold = sum(all_runs) / len(all_runs)
            hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)

        return float(freq_score + hold_bonus)

    def _turnover_penalty(self, turnover: Tensor) -> Tensor:
        """梯度式换手率惩罚。"""
        mean_to = turnover.mean()
        penalty = torch.clamp(
            (mean_to - 0.2) * 3.0,
            min=torch.tensor(0.0),
            max=torch.tensor(3.0),
        )
        return -penalty

    # ──────────────────────────────────────────────────────────────────────
    # Walk-Forward 辅助接口
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_fold(
        self,
        factors:     Tensor,
        target_ret:  Tensor,
        train_start: int,
        train_end:   int,
        val_start:   int,
        val_end:     int,
    ) -> tuple[Tensor, Tensor]:
        """在指定训练/验证切片上计算组合多目标得分。

        train_score 和 val_score 使用同一套多目标框架，保证"选王"标准一致。
        """
        position = compute_target_positions_stateless(factors)  # neutral band

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * self.cost_rate

        pnl_train = pnl[:, train_start:train_end]
        pnl_val   = pnl[:, val_start:val_end]

        # 训练段：多目标 + 换手率惩罚
        train_score = self._multi_objective(
            factors[:, train_start:train_end],
            target_ret[:, train_start:train_end],
            pnl_train,
            position[:, train_start:train_end],
        ) + self._turnover_penalty(turnover[:, train_start:train_end])

        # 验证段：同一套多目标（不加换手惩罚，避免双重惩罚）
        val_score = self._multi_objective(
            factors[:, val_start:val_end],
            target_ret[:, val_start:val_end],
            pnl_val,
            position[:, val_start:val_end],
        )

        return train_score, val_score

    def _multi_objective(
        self,
        factors:    Tensor,
        target_ret: Tensor,
        pnl:        Tensor,
        position:   Tensor,
    ) -> Tensor:
        """统一的多目标评分。

        N=1 时（单品种训练模式）：跳过多品种统计，直接用 Sortino+Calmar+IC+hold_quality。
        N>1 时（组合模式）：加入 symbol_consistency 和 cost_stress。
        """
        N = pnl.shape[0]
        port_sortino = self._sortino(pnl)
        port_calmar  = self._calmar(pnl)
        ts_ic        = self._ts_ic_stability(factors, target_ret)
        tq           = self._turnover_quality(position)

        if N == 1:
            # 单品种：不做 symbol_consistency 和 cost_stress（无意义）
            # 权重重新分配给 Sortino 和 IC
            return (
                0.45 * port_sortino
                + 0.25 * port_calmar
                + 0.20 * ts_ic
                + 0.10 * tq
            )

        # 多品种：完整多目标
        per_sym_sortino     = []
        per_sym_trade_count = []
        for n in range(N):
            per_sym_sortino.append(self._sortino(pnl[n]).item())
            pos_n = position[n].tolist()
            trades, prev = 0, 0
            for v in pos_n:
                vi = int(v)
                if vi != 0 and vi != prev:
                    trades += 1
                prev = vi if vi != 0 else prev
            per_sym_trade_count.append(trades)

        sym_cons = self._symbol_consistency(per_sym_sortino, per_sym_trade_count)
        cost_s   = self._cost_stress(position, target_ret)

        return (
            0.35 * port_sortino
            + 0.20 * port_calmar
            + 0.15 * ts_ic
            + 0.10 * sym_cons
            + 0.10 * cost_s
            + 0.10 * tq
        )

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口（非 Walk-Forward 模式）
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        factors:    Tensor,
        raw_dict:   dict,
        target_ret: Tensor,
    ) -> tuple[Tensor, float]:
        """评估一组 Alpha 因子（组合多目标得分）。"""
        position = compute_target_positions_stateless(factors)

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * self.cost_rate

        T     = factors.shape[1]
        split = int(math.floor(T * 0.8))

        score = self._multi_objective(
            factors[:, :split], target_ret[:, :split],
            pnl[:, :split], position[:, :split],
        ) + self._turnover_penalty(turnover[:, :split])

        mean_oos = pnl[:, split:].mean().item()
        return score, mean_oos
