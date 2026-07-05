"""
test_max_hold.py — 测试强制限制最大持仓时间对 Sharpe 的影响

用真实 forex 因子（best_forex.json）在 EURUSD+USDJPY 全量历史上：
  1. 基线：无持仓限制
  2. 约束：最大持仓 24h（持仓满 24 根 H1 后强制平仓 1 根，再视信号决定是否重进）
对比 Sharpe / Sortino / TotRet / MDD / AvgHold / Trades。
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

H1_PER_YEAR = 6240
COST_RATE = 0.0001
MAX_HOLD_BARS = 24   # 24 小时
FOREX_SYMS = ["EURUSD", "USDJPY"]


def calc_sharpe(pnl):
    flat = np.asarray(pnl).reshape(-1)
    std = flat.std()
    if std < 1e-8:
        return 0.0
    return float(flat.mean() / std * math.sqrt(H1_PER_YEAR))


def calc_sortino(pnl):
    flat = np.asarray(pnl).reshape(-1)
    downside = flat[flat < 0]
    if len(downside) == 0:
        return 0.0
    ds = downside.std()
    full_std = max(flat.std(), 1e-8)
    floor = max(full_std * 0.2, 1e-8)
    ds = max(ds, floor)
    return float(flat.mean() / ds * math.sqrt(H1_PER_YEAR))


def calc_mdd(pnl):
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def calc_metrics(pnl, position):
    """计算一组指标。pnl/position: [N, T] tensor → 按 N 等权组合。"""
    port_pnl = pnl.mean(dim=0).numpy()
    sharpe = calc_sharpe(port_pnl)
    sortino = calc_sortino(port_pnl)
    tot_ret = float(port_pnl.sum())
    mdd = calc_mdd(port_pnl)

    # 持仓时间统计
    pos_np = position.abs().mean(dim=0).numpy()  # 组合平均仓位
    # 用 sign 变化统计交易次数和持仓时长
    pos_sign = np.sign(port_pnl)
    runs, cur_len, cur_dir = [], 0, 0
    for p in pos_sign:
        d = int(p)
        if d == 0:
            if cur_len > 0:
                runs.append(cur_len)
            cur_dir, cur_len = 0, 0
        elif d == cur_dir:
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append(cur_len)
            cur_dir, cur_len = d, 1
    if cur_len > 0:
        runs.append(cur_len)

    n_trades = len(runs)
    avg_hold = sum(runs) / n_trades if n_trades else 0
    exposure = float(position.abs().mean())

    return {
        "sharpe":   sharpe,
        "sortino":  sortino,
        "tot_ret":  tot_ret,
        "mdd":      mdd,
        "n_trades": n_trades,
        "avg_hold_h": avg_hold,
        "exposure": exposure,
    }


def apply_max_hold(position: torch.Tensor, max_hold: int) -> torch.Tensor:
    """强制最大持仓时间：同方向持仓满 max_hold 根后，下一根强制平仓。

    平仓后下一根可立即根据信号重新进场（最宽松的解释）。
    """
    pos = position.clone()
    N, T = pos.shape
    for n in range(N):
        hold = 0
        cur_dir = 0
        for t in range(T):
            p = pos[n, t].item()
            d = 1 if p > 0.001 else (-1 if p < -0.001 else 0)
            if d == 0:
                hold = 0
                cur_dir = 0
                continue
            if d == cur_dir:
                hold += 1
                if hold >= max_hold:
                    # 持满 max_hold 根，强制平仓
                    pos[n, t] = 0.0
                    hold = 0
                    cur_dir = 0
            else:
                cur_dir = d
                hold = 1
    return pos


def apply_max_hold_strict(position: torch.Tensor, max_hold: int) -> torch.Tensor:
    """严格版：平仓后必须等信号反转才能重新进场。"""
    pos = position.clone()
    N, T = pos.shape
    for n in range(N):
        hold = 0
        cur_dir = 0
        locked_dir = 0  # 锁定方向，必须反向才解锁
        for t in range(T):
            p = pos[n, t].item()
            d = 1 if p > 0.001 else (-1 if p < -0.001 else 0)

            if locked_dir != 0 and d == locked_dir:
                # 信号未反转，继续空仓
                pos[n, t] = 0.0
                continue
            elif locked_dir != 0 and d != 0 and d != locked_dir:
                # 信号反转，解锁
                locked_dir = 0

            if d == 0:
                hold = 0
                cur_dir = 0
                continue
            if d == cur_dir:
                hold += 1
                if hold >= max_hold:
                    pos[n, t] = 0.0
                    hold = 0
                    cur_dir = 0
                    locked_dir = d  # 锁定，等反向信号
            else:
                cur_dir = d
                hold = 1
    return pos


def main():
    offline = "--offline" in sys.argv

    print(f"\n{'='*72}")
    print(f"  最大持仓时间影响测试  |  max_hold = {MAX_HOLD_BARS}h")
    print(f"{'='*72}\n")

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
    Config.SYMBOLS = FOREX_SYMS
    try:
        with MT5DataFetcher(offline=offline) as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            raw_dict = mgr.raw_dict
            target_ret = mgr.target_ret
            T = raw_dict["open"].shape[1]
            print(f"数据: {mgr.symbols}  T={T} bars\n")

            feat = MT5FeatureEngineer.compute_features(raw_dict)

            # ── 3. 计算因子值 ───────────────────────────────────────
            vm = StackVM()
            factor = vm.execute(formula, feat)
            if factor is None:
                print("[ERROR] 因子执行失败"); return

            # ── 4. 基线回测（无限制）──────────────────────────────
            pos_base = compute_target_positions_stateless(factor)
            prev_pos = torch.roll(pos_base, 1, dims=1)
            prev_pos[:, 0] = 0.0
            turnover_base = torch.abs(pos_base - prev_pos)
            pnl_base = pos_base * target_ret - turnover_base * COST_RATE

            m_base = calc_metrics(pnl_base, pos_base)

            # ── 5. 约束回测（max_hold=24h，宽松版）─────────────────
            pos_constrained = apply_max_hold(pos_base, MAX_HOLD_BARS)
            prev_pos_c = torch.roll(pos_constrained, 1, dims=1)
            prev_pos_c[:, 0] = 0.0
            turnover_c = torch.abs(pos_constrained - prev_pos_c)
            pnl_c = pos_constrained * target_ret - turnover_c * COST_RATE
            m_c = calc_metrics(pnl_c, pos_constrained)

            # ── 6. 约束回测（max_hold=24h，严格版）─────────────────
            pos_strict = apply_max_hold_strict(pos_base, MAX_HOLD_BARS)
            prev_pos_s = torch.roll(pos_strict, 1, dims=1)
            prev_pos_s[:, 0] = 0.0
            turnover_s = torch.abs(pos_strict - prev_pos_s)
            pnl_s = pos_strict * target_ret - turnover_s * COST_RATE
            m_s = calc_metrics(pnl_s, pos_strict)

    finally:
        Config.SYMBOLS = original_symbols

    # ── 7. 对比表 ───────────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"  结果对比（EURUSD+USDJPY 等权组合，{T} bars H1）")
    print(f"{'='*72}")
    hdr = f"  {'模式':22s} {'Sharpe':>7} {'Sortino':>8} {'TotRet':>8} {'MDD':>7} {'AvgHold':>8} {'Trades':>7} {'Exposure':>8}"
    print(hdr)
    print(f"  {'-'*82}")

    rows = [
        ("基线（无限制）",       m_base),
        (f"max_hold=24h（宽松）", m_c),
        (f"max_hold=24h（严格）", m_s),
    ]
    for label, m in rows:
        print(f"  {label:22s} "
              f"{m['sharpe']:>+7.3f} "
              f"{m['sortino']:>+8.3f} "
              f"{m['tot_ret']:>+8.3f} "
              f"{m['mdd']:>7.3f} "
              f"{m['avg_hold_h']:>7.1f}h "
              f"{m['n_trades']:>7d} "
              f"{m['exposure']:>7.1%}")

    # ── 8. Sharpe 变化 ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Sharpe 变化分析")
    print(f"{'='*72}")
    base_sharpe = m_base["sharpe"]
    for label, m in rows[1:]:
        delta = m["sharpe"] - base_sharpe
        pct = (delta / abs(base_sharpe) * 100) if base_sharpe != 0 else 0
        print(f"  {label:22s}: Sharpe {base_sharpe:+.3f} → {m['sharpe']:+.3f}  "
              f"(Δ={delta:+.3f}, {pct:+.1f}%)")

    # ── 9. 不同 max_hold 的敏感性扫描 ───────────────────────────────
    print(f"\n{'='*72}")
    print(f"  敏感性扫描：不同 max_hold 下的 Sharpe")
    print(f"{'='*72}")
    print(f"  {'max_hold':>8} {'Sharpe':>7} {'Sortino':>8} {'TotRet':>8} {'AvgHold':>8} {'Trades':>7}")
    print(f"  {'-'*52}")

    Config.SYMBOLS = FOREX_SYMS
    try:
        with MT5DataFetcher(offline=offline) as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            raw_dict = mgr.raw_dict
            target_ret = mgr.target_ret
            feat = MT5FeatureEngineer.compute_features(raw_dict)
            vm = StackVM()
            factor = vm.execute(formula, feat)
            pos_base = compute_target_positions_stateless(factor)

            for mh in [6, 12, 24, 48, 96, 240, 999999]:
                pos_c = apply_max_hold(pos_base, mh)
                prev = torch.roll(pos_c, 1, dims=1)
                prev[:, 0] = 0.0
                to = torch.abs(pos_c - prev)
                pnl = pos_c * target_ret - to * COST_RATE
                m = calc_metrics(pnl, pos_c)
                label = f"{mh}h" if mh < 999999 else "无限制"
                print(f"  {label:>8} {m['sharpe']:>+7.3f} {m['sortino']:>+8.3f} "
                      f"{m['tot_ret']:>+8.3f} {m['avg_hold_h']:>7.1f}h {m['n_trades']:>7d}")
    finally:
        Config.SYMBOLS = original_symbols

    print(f"\n完成。\n")


if __name__ == "__main__":
    main()
