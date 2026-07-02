"""
run_backtest.py — 独立回测运行脚本

用法：
    python run_backtest.py

流程：
    1. 连接 MT5，拉取最新 OHLCV 数据
    2. 用 best_mt5_strategy.json 中的公式跑 BacktestEngine
    3. 生成全局 K 线图（最近 120 根）
    4. 为每笔交易生成局部缩放图
    5. 保存 JSON 报告并打印统计摘要
"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from backtest_viz import BacktestEngine, BacktestChart, BacktestReport
from model_core.vocab import FORMULA_VOCAB


def decode_formula(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(
        names[t] if 0 <= t < len(names) else f"?{t}"
        for t in tokens
    )


def main():
    OUTPUT_DIR = "backtest_output"

    # ── 1. 加载最优公式 ───────────────────────────────────────────────
    strategy_path = Path(Config.STRATEGY_FILE)
    if not strategy_path.exists():
        print(f"[ERROR] 找不到策略文件: {strategy_path}")
        sys.exit(1)

    with open(strategy_path) as f:
        data = json.load(f)

    # 支持新格式（含 vocab_version）和旧格式（纯列表）
    if isinstance(data, dict):
        formula = data["formula"]
        vocab_ver = data.get("vocab_version", "unknown")
        print(f"  策略词表版本     : {vocab_ver}")
    else:
        formula = data
        vocab_ver = "legacy"

    formula_readable = decode_formula(formula)
    print(f"\n{'='*60}")
    print(f"  公式 tokens : {formula}")
    print(f"  公式解读    : {formula_readable}")
    print(f"{'='*60}\n")

    # ── 2. 连接 MT5，加载数据 ─────────────────────────────────────────
    print("正在连接 MT5 并拉取数据...")
    with MT5DataFetcher() as fetcher:
        data_mgr = MT5DataManager(fetcher)
        data_mgr.load()

        raw_dict    = data_mgr.raw_dict
        feat_tensor = data_mgr.feat_tensor
        symbols     = data_mgr.symbols

        T = raw_dict["open"].shape[1]
        print(f"  品种: {symbols}")
        print(f"  数据长度: T={T} bars\n")

        # ── 3. 运行回测引擎 ───────────────────────────────────────────
        print("运行回测引擎...")
        engine = BacktestEngine(
            formula=formula,
            cost_rate=Config.COST_RATE,
        )
        results = engine.run(raw_dict, feat_tensor, symbols)

    # ── 4. 打印统计报告 ───────────────────────────────────────────────
    report = BacktestReport(formula=formula, formula_readable=formula_readable)
    report.print_summary(results)

    # ── 5. 保存 JSON 报告 ─────────────────────────────────────────────
    report.save_json(results, path=f"{OUTPUT_DIR}/real_report.json")

    # ── 6. 生成图表 ───────────────────────────────────────────────────
    print("\n生成图表...")
    chart = BacktestChart(max_bars=120)

    chart.plot_all(results, output_dir=OUTPUT_DIR)

    for r in results:
        saved = chart.plot_all_trade_zooms(
            r,
            output_dir=OUTPUT_DIR,
            pre_bars=25,
            post_bars=12,
            max_trades=15,
        )
        print(f"  {r.symbol}: 共生成 {len(saved)} 张缩放图")

    # ── 7. 组合汇总 ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  组合级汇总")
    print(f"{'='*60}")
    total_pnl   = sum(r.total_return for r in results)
    n_positive  = sum(1 for r in results if r.total_return > 0)
    n_sortino_p = sum(1 for r in results if r.sortino > 0)
    avg_hold    = sum(r.avg_hold_bars for r in results) / len(results)
    print(f"  全品种总 PnL       : {total_pnl:+.4f}")
    print(f"  正收益品种数       : {n_positive}/{len(results)}")
    print(f"  Sortino>0 品种数   : {n_sortino_p}/{len(results)}")
    print(f"  全品种平均持仓     : {avg_hold:.1f} bars")
    best  = max(results, key=lambda r: r.sortino)
    worst = min(results, key=lambda r: r.sortino)
    print(f"  最佳品种           : {best.symbol} (Sortino={best.sortino:.2f})")
    print(f"  最差品种           : {worst.symbol} (Sortino={worst.sortino:.2f})")
    consistency_ok = n_sortino_p >= len(results) * 0.6
    no_disaster    = all(r.sortino >= -2.0 for r in results)
    print(f"  品种一致性(>=60%)  : {'PASS' if consistency_ok else 'FAIL'}")
    print(f"  无灾难亏损(>=-2)   : {'PASS' if no_disaster else 'FAIL'}")
    print(f"{'='*60}")
    print(f"\n全部图表已保存至 {OUTPUT_DIR}/")
    print("完成。\n")


if __name__ == "__main__":
    main()
