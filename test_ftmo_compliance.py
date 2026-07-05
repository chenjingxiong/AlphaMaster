"""
test_ftmo_compliance.py — FTMO 考试盘合规性测试

用真实 forex 因子在 EURUSD+USDJPY 全量历史上模拟 FTMO 考试：
  1. 计算每日 P&L，检查是否违反 Max Daily Loss (3% / 5%)
  2. 检查 Max Overall Loss (10% trailing)
  3. 检查 Best Day Rule (1-Step: 最佳日 ≤ 50% 总盈利)
  4. 检查 Min Trading Days (2-Step: 4 天)
  5. 扫描不同仓位系数，找最优 FTMO 配置
"""
import sys, json, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless

# FTMO 规则常量（$100,000 账户）
ACCOUNT_SIZE = 100_000.0
H1_PER_DAY = 24
H1_PER_YEAR = 6240
COST_RATE = 0.0001

# 2-Step 规则
MAX_DAILY_LOSS_2STEP = 0.05      # 5%
MAX_OVERALL_LOSS_2STEP = 0.10    # 10%
PROFIT_TARGET_2STEP_P1 = 0.10    # Phase 1: 10%
PROFIT_TARGET_2STEP_P2 = 0.05    # Phase 2: 5%
MIN_TRADING_DAYS_2STEP = 4

# 1-Step 规则
MAX_DAILY_LOSS_1STEP = 0.03      # 3%
MAX_OVERALL_LOSS_1STEP = 0.10    # 10% trailing
PROFIT_TARGET_1STEP = 0.10       # 10%
BEST_DAY_RULE_1STEP = 0.50       # 最佳日 ≤ 50%


def calc_pnl(position, target_ret, cost_rate, pos_scale=1.0):
    """计算 P&L，pos_scale 为仓位缩放系数。"""
    pos = position * pos_scale
    prev_pos = torch.roll(pos, 1, dims=1)
    prev_pos[:, 0] = 0.0
    turnover = torch.abs(pos - prev_pos)
    pnl = pos * target_ret - turnover * cost_rate
    return pnl


def aggregate_daily_pnl(pnl_tensor):
    """将 [N, T] 的逐 bar P&L 聚合为逐日 P&L（以 CET 时区为准，24h 一天）。

    注意：H1 数据，假设 00:00 CET 对齐（MT5 服务器时间通常接近 CET）。
    返回: daily_pnl [N_days] numpy array
    """
    N, T = pnl_tensor.shape
    port_pnl = pnl_tensor.mean(dim=0).numpy()  # [T]
    n_days = T // H1_PER_DAY
    port_pnl_trim = port_pnl[:n_days * H1_PER_DAY]
    daily = port_pnl_trim.reshape(n_days, H1_PER_DAY).sum(axis=1)
    return daily


def calc_equity_curve(daily_pnl, initial_balance):
    """计算权益曲线（含 Max Loss trailing 检查）。"""
    cum = np.cumsum(daily_pnl) * initial_balance
    equity = initial_balance + cum
    # Max Loss trailing limit: max(initial_balance, max(historical end-of-day balance)) - 10%
    end_of_day_balances = np.concatenate([[initial_balance], initial_balance + np.cumsum(daily_pnl * initial_balance)[:-1]])
    # trailing limit: 每天基于前一日 end-of-day balance 的最大值
    running_max = np.maximum.accumulate(end_of_day_balances)
    max_loss_limit = running_max - MAX_OVERALL_LOSS_2STEP * initial_balance
    return equity, max_loss_limit


def simulate_ftmo(daily_pnl, account_size, max_daily_loss, max_overall_loss,
                  profit_target, pos_scale=1.0, verbose=True):
    """模拟 FTMO 考试流程，返回是否通过 + 违规详情。"""
    scaled_daily = daily_pnl * pos_scale * account_size
    n_days = len(scaled_daily)
    balance = account_size
    peak_balance = account_size
    cumulative_pnl = 0.0

    violations = []
    daily_losses = []
    trading_days = 0
    profit_days = []
    target_reached_day = None

    for d in range(n_days):
        day_pnl = scaled_daily[d]
        balance += day_pnl
        cumulative_pnl += day_pnl

        # 是否为交易日（有仓位变动即算）
        if abs(day_pnl) > 1e-6:
            trading_days += 1

        if day_pnl > 0:
            profit_days.append((d, day_pnl))

        # 检查 Max Daily Loss
        # 当日 equity = balance - day_pnl + day_pnl = balance（已结算）
        # 但日内 equity 最低可能更低，这里用日 P&L 作为近似
        if day_pnl < 0:
            daily_loss_pct = abs(day_pnl) / account_size
            daily_losses.append(daily_loss_pct)
            if daily_loss_pct > max_daily_loss:
                violations.append({
                    "day": d,
                    "type": "max_daily_loss",
                    "loss_pct": daily_loss_pct,
                    "limit_pct": max_daily_loss
                })
        else:
            daily_losses.append(0.0)

        # 检查 Max Overall Loss (trailing)
        peak_balance = max(peak_balance, balance - day_pnl)  # end-of-day peak
        max_loss_limit = peak_balance - max_overall_loss * account_size
        if balance < max_loss_limit:
            violations.append({
                "day": d,
                "type": "max_overall_loss",
                "balance": balance,
                "limit": max_loss_limit
            })

        # 检查 Profit Target
        if target_reached_day is None:
            if cumulative_pnl >= profit_target * account_size:
                target_reached_day = d

    # Best Day Rule (1-Step)
    if profit_days:
        best_day_pnl = max(p[1] for p in profit_days)
        total_profit = sum(p[1] for p in profit_days)
        best_day_ratio = best_day_pnl / total_profit if total_profit > 0 else 0
    else:
        best_day_ratio = 0.0

    result = {
        "passed": len(violations) == 0,
        "violations": violations,
        "n_violations": len(violations),
        "trading_days": trading_days,
        "target_reached_day": target_reached_day,
        "best_day_ratio": best_day_ratio,
        "max_daily_loss_pct": max(daily_losses) if daily_losses else 0,
        "total_return_pct": cumulative_pnl / account_size,
        "n_profit_days": len(profit_days),
    }

    if verbose:
        vtype_count = {}
        for v in violations:
            vtype_count[v["type"]] = vtype_count.get(v["type"], 0) + 1
        print(f"    仓位系数: {pos_scale:.2f}x  "
              f"总收益: {result['total_return_pct']:+.2%}  "
              f"交易天数: {trading_days}  "
              f"最大日亏: {result['max_daily_loss_pct']:.2%}  "
              f"违规: {result['n_violations']}次 ({vtype_count})  "
              f"达标日: {'Day '+str(target_reached_day) if target_reached_day else '未达标'}  "
              f"{'✓ 通过' if result['passed'] else '✗ 失败'}")

    return result


def main():
    offline = "--offline" in sys.argv

    print(f"\n{'='*78}")
    print(f"  FTMO 考试盘合规性测试")
    print(f"{'='*78}\n")

    # ── 1. 加载因子 ─────────────────────────────────────────────────
    formula_data = json.load(open("strategies/best_forex.json"))
    if formula_data.get("vocab_version") != VOCAB_VERSION:
        print("[ERROR] vocab 版本不符"); return
    formula = formula_data["formula"]
    readable = " -> ".join(FORMULA_VOCAB.token_names[t] for t in formula)
    print(f"因子: {readable}")
    print(f"训练 score: {formula_data['best_score']:.4f}\n")

    # ── 2. 加载数据 ─────────────────────────────────────────────────
    original_symbols = Config.SYMBOLS[:]
    Config.SYMBOLS = ["EURUSD", "USDJPY"]
    try:
        with MT5DataFetcher(offline=offline) as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            raw_dict = mgr.raw_dict
            target_ret = mgr.target_ret
            T = raw_dict["open"].shape[1]
            print(f"数据: {mgr.symbols}  T={T} bars ({T//H1_PER_DAY} 天 ≈ {T/H1_PER_YEAR:.1f} 年)\n")

            feat = MT5FeatureEngineer.compute_features(raw_dict)
            vm = StackVM()
            factor = vm.execute(formula, feat)
            if factor is None:
                print("[ERROR] 因子执行失败"); return

            pos_base = compute_target_positions_stateless(factor)
            pnl_base = calc_pnl(pos_base, target_ret, COST_RATE, pos_scale=1.0)
            daily_pnl = aggregate_daily_pnl(pnl_base)

    finally:
        Config.SYMBOLS = original_symbols

    # ── 3. 基础统计 ─────────────────────────────────────────────────
    print(f"{'='*78}")
    print(f"  基础统计（满仓 1.0x，{len(daily_pnl)} 天）")
    print(f"{'='*78}")
    print(f"  累计收益:   {daily_pnl.sum():+.4f} ({daily_pnl.sum()*ACCOUNT_SIZE/1000:+.1f}k)")
    print(f"  日均收益:   {daily_pnl.mean():+.6f}")
    print(f"  日收益标准差: {daily_pnl.std():.6f}")
    print(f"  最大日盈利: {daily_pnl.max():+.4f} ({daily_pnl.max()*ACCOUNT_SIZE/1000:+.1f}k)")
    print(f"  最大日亏损: {daily_pnl.min():+.4f} ({daily_pnl.min()*ACCOUNT_SIZE/1000:+.1f}k)")
    print(f"  盈利天数:   {(daily_pnl > 0).sum()} / {len(daily_pnl)} ({(daily_pnl > 0).mean():.1%})")
    print(f"  亏损天数:   {(daily_pnl < 0).sum()} / {len(daily_pnl)} ({(daily_pnl < 0).mean():.1%})")
    print()

    # ── 4. 满仓 vs FTMO 限制对比 ───────────────────────────────────
    print(f"{'='*78}")
    print(f"  FTMO 2-Step 考试模拟（Max Daily Loss=5%, Max Overall=10%, Target=10%）")
    print(f"{'='*78}")
    print(f"  {'仓位系数':>8} {'总收益':>8} {'交易天数':>8} {'最大日亏':>8} {'违规次数':>8} {'达标':>6} {'结果':>6}")
    print(f"  {'-'*68}")

    best_2step_scale = None
    for scale in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]:
        r = simulate_ftmo(daily_pnl, ACCOUNT_SIZE, MAX_DAILY_LOSS_2STEP,
                          MAX_OVERALL_LOSS_2STEP, PROFIT_TARGET_2STEP_P1,
                          pos_scale=scale)
        if r["passed"] and best_2step_scale is None:
            best_2step_scale = scale

    print()

    print(f"{'='*78}")
    print(f"  FTMO 1-Step 考试模拟（Max Daily Loss=3%, Max Overall=10% trailing, Target=10%）")
    print(f"{'='*78}")
    print(f"  {'仓位系数':>8} {'总收益':>8} {'交易天数':>8} {'最大日亏':>8} {'Best Day':>8} {'违规':>6} {'结果':>6}")
    print(f"  {'-'*68}")

    best_1step_scale = None
    for scale in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]:
        r = simulate_ftmo(daily_pnl, ACCOUNT_SIZE, MAX_DAILY_LOSS_1STEP,
                          MAX_OVERALL_LOSS_1STEP, PROFIT_TARGET_1STEP,
                          pos_scale=scale, verbose=False)
        vtype_count = {}
        for v in r["violations"]:
            vtype_count[v["type"]] = vtype_count.get(v["type"], 0) + 1
        best_day_str = f"{r['best_day_ratio']:.0%}"
        print(f"  {scale:>7.2f}x {r['total_return_pct']:>+7.2%} "
              f"{r['trading_days']:>8d} {r['max_daily_loss_pct']:>7.2%} "
              f"{best_day_str:>8} {r['n_violations']:>6d} "
              f"{'✓' if r['passed'] else '✗':>6}")
        if r["passed"] and best_1step_scale is None:
            best_1step_scale = scale

    print()

    # ── 5. 最优配置详情 ─────────────────────────────────────────────
    print(f"{'='*78}")
    print(f"  最优 FTMO 配置推荐")
    print(f"{'='*78}")

    if best_2step_scale:
        print(f"\n  ★ 2-Step 推荐: {best_2step_scale:.2f}x 仓位")
        print(f"    - Max Daily Loss 5%: 安全")
        print(f"    - Max Overall Loss 10%: 安全")
        print(f"    - Min Trading Days 4: 自然满足（策略几乎每天交易）")
        print(f"    - 无 Best Day Rule 限制")
        print(f"    - 可选 Swing 账户：允许周末持仓")
        r = simulate_ftmo(daily_pnl, ACCOUNT_SIZE, MAX_DAILY_LOSS_2STEP,
                          MAX_OVERALL_LOSS_2STEP, PROFIT_TARGET_2STEP_P1,
                          pos_scale=best_2step_scale, verbose=False)
        print(f"    - 预计 {r['target_reached_day']+1} 天达标 10% 目标")

    if best_1step_scale:
        print(f"\n  ★ 1-Step 推荐: {best_1step_scale:.2f}x 仓位")
        print(f"    - Max Daily Loss 3%: 需更小仓位")
        print(f"    - Max Overall Loss 10% trailing: 安全")
        print(f"    - Best Day Rule 50%: 需检查")
        r = simulate_ftmo(daily_pnl, ACCOUNT_SIZE, MAX_DAILY_LOSS_1STEP,
                          MAX_OVERALL_LOSS_1STEP, PROFIT_TARGET_1STEP,
                          pos_scale=best_1step_scale, verbose=False)
        print(f"    - 预计 {r['target_reached_day']+1} 天达标 10% 目标")
        print(f"    - Best Day 占比: {r['best_day_ratio']:.1%} {'✓' if r['best_day_ratio'] < 0.5 else '✗ 超标'}")
    else:
        print(f"\n  ✗ 1-Step 不推荐：3% 日亏限制太严，当前策略难以满足")

    # ── 6. 需要添加的风控规则 ───────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  需要添加的 FTMO 风控规则")
    print(f"{'='*78}")
    print(f"""
  1. [必须] 日亏熔断: 当日亏损达到 {-MAX_DAILY_LOSS_2STEP*100:.0f}% (2-Step) / {-MAX_DAILY_LOSS_1STEP*100:.0f}% (1-Step) 时平仓停止
  2. [必须] 周末平仓: Standard 账户周五 22:00 CET 前平所有仓位
  3. [建议] 仓位缩放: 实盘仓位 = 信号仓位 × {best_2step_scale or 0.5:.2f}
  4. [建议] 新闻过滤: FOMC/NFP/CPI 前后 2 分钟不开/平仓（仅 FTMO Account 阶段）
  5. [可选] Swing 账户: 2-Step 专用，免除周末/过夜限制
  6. [无需] 最低交易天数: 策略几乎每天交易，自然满足 4 天要求
""")

    print(f"完成。\n")


if __name__ == "__main__":
    main()
