"""
strategy_manager/runner.py — MT5 策略主循环控制器（回测对标版）

核心改动（vs 旧版本）：
  1. 信号改为 tanh→sign，与 backtest.py 完全一致（Config.SIGNAL_MODE）
  2. 入场/出场统一为「信号翻转驱动」(_reconcile_positions)
  3. 支持做空，多/空均可反手
  4. K 线收盘触发调仓（REBALANCE_ON_BAR_CLOSE=True），消除时间偏差
  5. EXIT_MODE 控制是否叠加风控层（signal / risk / hybrid）
  6. MAX_OPEN_POSITIONS=None 表示不限制，严格对标回测
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

    class _MT5Stub:
        def shutdown(self) -> None:
            pass

    mt5 = _MT5Stub()  # type: ignore[assignment]

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from execution.price_feed import MT5PriceFeed
from execution.trader import MT5Trader
from model_core.vm import StackVM
from strategy_manager.portfolio import MT5PortfolioManager
from strategy_manager.risk import MT5RiskEngine
from strategy_manager.signal import (
    compute_target_positions,
    reconcile_action,
    HOLD, OPEN_LONG, OPEN_SHORT, CLOSE, REVERSE_TO_LONG, REVERSE_TO_SHORT,
)

_LOOP_INTERVAL: int = 60


class MT5StrategyRunner:
    """同步策略主循环控制器（回测对标版）。

    与旧版本关键差异：
    - 使用 compute_target_positions() 替代 sigmoid+阈值
    - _reconcile_positions() 替代 _scan_for_entries()
    - K 线收盘触发，消除回测-实盘时间偏差
    - 支持做空与反手
    - EXIT_MODE 控制风控叠加
    """

    def __init__(self) -> None:
        strategy_path = Config.STRATEGY_FILE
        if not os.path.exists(strategy_path):
            logger.critical(
                f"Strategy file '{strategy_path}' not found. "
                "Please run main.py to train first."
            )
            sys.exit(1)

        try:
            with open(strategy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.critical(f"Failed to load strategy '{strategy_path}': {exc}")
            sys.exit(1)

        if isinstance(data, list):
            logger.critical(
                "[Runner] Legacy strategy format (plain list) is not supported "
                "after vocab v2.0. Feature order has changed from 10 to 20 dims. "
                "Please retrain with the current feature set (python main.py)."
            )
            sys.exit(1)
        elif isinstance(data, dict) and "formula" in data:
            self.formula = [int(t) for t in data["formula"]]
            # vocab_version 校验：不一致则拒绝启动，防止特征语义错位
            from model_core.vocab import VOCAB_VERSION as _CURRENT_VER
            strategy_ver = data.get("vocab_version", "unknown")
            if strategy_ver != _CURRENT_VER:
                logger.critical(
                    f"[Runner] vocab_version mismatch: "
                    f"strategy={strategy_ver}, current={_CURRENT_VER}. "
                    f"Please retrain with the current feature set."
                )
                sys.exit(1)
        else:
            logger.critical(f"Unexpected strategy format in '{strategy_path}'.")
            sys.exit(1)

        logger.success(f"[Runner] Strategy loaded: {self.formula}")

        self.vm        = StackVM()
        self.portfolio = MT5PortfolioManager()
        self.risk      = MT5RiskEngine()
        self.trader    = MT5Trader()

        self._fetcher: MT5DataFetcher | None       = None
        self._data_manager: MT5DataManager | None  = None
        self._last_refresh: float                   = 0.0
        # K 线收盘检测：记录上次调仓时的 bar_time，形状 [N]
        self._last_bar_time: torch.Tensor | None    = None


    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """同步主循环。

        流程：
            1. 连接 MT5 终端
            2. while True:
               a. 检查停止信号
               b. 按需刷新数据
               c. 若 REBALANCE_ON_BAR_CLOSE=True，只在新 K 线收盘时调仓
               d. 同步 MT5 仓位
               e. 调仓（_reconcile_positions）
               f. 若 EXIT_MODE in ('risk','hybrid')，叠加风控监控
               g. 休眠
        """
        logger.info("[Runner] Starting MT5StrategyRunner (backtest-parity mode)...")
        logger.info(f"  SIGNAL_MODE={Config.SIGNAL_MODE}  EXIT_MODE={Config.EXIT_MODE}  "
                    f"MAX_OPEN_POSITIONS={Config.MAX_OPEN_POSITIONS}  "
                    f"REBALANCE_ON_BAR_CLOSE={Config.REBALANCE_ON_BAR_CLOSE}")

        try:
            self.trader.connect()
        except (ConnectionError, RuntimeError) as exc:
            logger.critical(f"[Runner] Cannot connect MT5 trader: {exc}")
            sys.exit(1)

        self._fetcher = MT5DataFetcher()
        try:
            self._fetcher.connect()
        except ConnectionError as exc:
            logger.critical(f"[Runner] Cannot connect MT5 fetcher: {exc}")
            sys.exit(1)

        self._data_manager = MT5DataManager(self._fetcher)
        try:
            self._data_manager.load()
            self._last_refresh = time.time()
        except Exception as exc:
            logger.error(f"[Runner] Initial data load failed: {exc}")

        logger.info("[Runner] MT5 connections established. Entering main loop.")

        while True:
            loop_start = time.time()

            # a. 停止信号
            if self._handle_stop_signal():
                logger.info("[Runner] Stop signal detected. Exiting.")
                break

            # b. 数据刷新
            if time.time() - self._last_refresh >= Config.DATA_REFRESH_INTERVAL:
                try:
                    self._data_manager.reload()
                    self._last_refresh = time.time()
                    logger.info("[Runner] Data refreshed.")
                except Exception as exc:
                    logger.error(f"[Runner] Data reload failed: {exc}")

            # c. 检测新 K 线收盘
            new_bar = True
            if Config.REBALANCE_ON_BAR_CLOSE and self._data_manager is not None:
                try:
                    cur_bar_time = self._data_manager.bar_time   # [N]
                    if (self._last_bar_time is not None and
                            cur_bar_time.shape == self._last_bar_time.shape and
                            (cur_bar_time == self._last_bar_time).all()):
                        new_bar = False
                    else:
                        self._last_bar_time = cur_bar_time.clone()
                except Exception as exc:
                    logger.warning(f"[Runner] bar_time check failed: {exc}")

            # d. 同步 MT5 仓位
            try:
                self.portfolio.sync_from_mt5()
            except Exception as exc:
                logger.warning(f"[Runner] Portfolio sync failed: {exc}")

            if new_bar:
                # e. 计算信号并对账调仓
                targets = self._compute_targets()
                if targets is not None:
                    try:
                        self._reconcile_positions(targets)
                    except Exception as exc:
                        logger.error(f"[Runner] _reconcile_positions raised: {exc}")
            else:
                logger.debug("[Runner] Same bar, skipping rebalance.")

            # f. 风控监控（可选叠加层）
            if Config.EXIT_MODE in ("risk", "hybrid"):
                try:
                    self._monitor_positions()
                except Exception as exc:
                    logger.error(f"[Runner] _monitor_positions raised: {exc}")

            # g. 休眠
            elapsed = time.time() - loop_start
            sleep_t = max(10, _LOOP_INTERVAL - elapsed)
            logger.info(f"[Runner] Cycle {elapsed:.2f}s. Sleep {sleep_t:.2f}s.")
            time.sleep(sleep_t)

    def shutdown(self) -> None:
        logger.info("[Runner] Shutting down...")
        try:
            if self._fetcher is not None:
                self._fetcher.shutdown()
        except Exception as exc:
            logger.warning(f"[Runner] fetcher.shutdown() raised: {exc}")
        mt5.shutdown()
        logger.info("[Runner] Stopped.")


    # ──────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────

    def _handle_stop_signal(self) -> bool:
        stop_path = Config.STOP_SIGNAL
        if not os.path.exists(stop_path):
            return False
        logger.warning(f"[Runner] STOP_SIGNAL detected at '{stop_path}'.")
        try:
            with open(stop_path, "w", encoding="utf-8") as f:
                f.write("STOPPED")
        except OSError as exc:
            logger.warning(f"[Runner] Failed to mark stop signal: {exc}")
        return True

    def _compute_targets(self) -> torch.Tensor | None:
        """运行 StackVM 并返回各品种目标仓位 {-1, 0, +1}，形状 [N]。

        实盘使用 stateful neutral band：传入当前持仓方向，
        中间区 [EXIT_THRESHOLD, ENTRY_THRESHOLD] 保持前仓（滞后出场）。
        这与回测（stateless，中间区直接空仓）有微小差异：
        回测保守，实盘稍宽松，是有意设计的安全边际。
        """
        if self._data_manager is None:
            return None
        try:
            feat = self._data_manager.feat_tensor      # [N, F, T]
            raw  = self.vm.execute(self.formula, feat)  # [N, T] or None
            if raw is None:
                logger.error("[Runner] StackVM returned None.")
                return None

            latest = raw[:, -1]   # [N]，最新 bar 的因子值

            if Config.SIGNAL_MODE == "backtest_parity":
                # 组装当前持仓方向 tensor，用于 neutral band 滞后出场
                symbols = self._data_manager.symbols
                prev_dirs = torch.zeros(len(symbols), dtype=torch.float32)
                for i, sym in enumerate(symbols):
                    prev_dirs[i] = float(self.portfolio.get_direction(sym))

                targets = compute_target_positions(latest, prev_positions=prev_dirs)
            else:
                # threshold 模式（旧逻辑，仅做多）
                scores  = torch.sigmoid(latest)
                targets = torch.zeros(len(scores), dtype=torch.float32)
                targets[scores > Config.BUY_THRESHOLD]  =  1.0
                targets[scores < Config.SELL_THRESHOLD] = -1.0

            return targets.float()
        except Exception as exc:
            logger.error(f"[Runner] _compute_targets failed: {exc}")
            return None

    def _reconcile_positions(self, targets: torch.Tensor) -> None:
        """对每个品种对账并执行调仓（替代旧版 _scan_for_entries）。

        对账逻辑（严格对标回测）：
            current = portfolio.get_direction(symbol)  # +1 / -1 / 0
            target  = targets[i]                       # +1 / -1 / 0
            action  = reconcile_action(current, target)

        根据 action 执行对应 MT5 订单。
        """
        if self._data_manager is None:
            return

        symbols = self._data_manager.symbols
        n = min(len(symbols), len(targets))

        for idx in range(n):
            symbol  = symbols[idx]
            target  = int(targets[idx].item())   # +1 / -1 / 0
            current = self.portfolio.get_direction(symbol)
            action  = reconcile_action(current, target)

            if action == HOLD:
                logger.debug(f"[Reconcile] {symbol}: HOLD (dir={current})")
                continue

            # MAX_OPEN_POSITIONS 约束（None 表示不限）
            max_pos = Config.MAX_OPEN_POSITIONS
            if max_pos is not None and action in (OPEN_LONG, OPEN_SHORT):
                if self.portfolio.get_open_count() >= max_pos:
                    logger.info(
                        f"[Reconcile] {symbol}: skip {action} — "
                        f"max_positions={max_pos} reached"
                    )
                    continue

            logger.info(f"[Reconcile] {symbol}: {action}  current={current}→target={target}")

            # 计算手数
            lot = self._calc_lot(symbol)
            if lot <= 0:
                logger.warning(f"[Reconcile] {symbol}: lot=0, skipping.")
                continue

            pos = self.portfolio.positions.get(symbol)
            ticket = pos.ticket if pos is not None else 0

            # ── 执行动作 ────────────────────────────────────────────
            if action == OPEN_LONG:
                ok = self.trader.buy(symbol, lot)
                if ok:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "BUY")

            elif action == OPEN_SHORT:
                ok = self.trader.open_short(symbol, lot)
                if ok:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "SELL")

            elif action == CLOSE:
                direction = pos.direction if pos else "BUY"
                ok = self.trader.close_position(symbol, pos.lot_size if pos else lot,
                                                direction, ticket)
                if ok:
                    self.portfolio.close_position(symbol)

            elif action == REVERSE_TO_LONG:
                # 先平空，再开多
                if pos:
                    ok_close = self.trader.close_position(
                        symbol, pos.lot_size, pos.direction, ticket
                    )
                    if ok_close:
                        self.portfolio.close_position(symbol)
                ok_open = self.trader.buy(symbol, lot)
                if ok_open:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "BUY")

            elif action == REVERSE_TO_SHORT:
                # 先平多，再开空
                if pos:
                    ok_close = self.trader.close_position(
                        symbol, pos.lot_size, pos.direction, ticket
                    )
                    if ok_close:
                        self.portfolio.close_position(symbol)
                ok_open = self.trader.open_short(symbol, lot)
                if ok_open:
                    price = self._get_price(symbol)
                    self.portfolio.add_position(symbol, 0, price, lot, "SELL")

    def _monitor_positions(self) -> None:
        """可选风控层（EXIT_MODE='risk' 或 'hybrid'）。

        多头：profit = current/entry - 1
        空头：profit = entry/current - 1（方向相反）
        止损、部分止盈、追踪止损逻辑同旧版，但空头追踪最低价。

        hybrid 模式下仅做紧急熔断（止损），不做部分止盈/追踪止损。
        """
        for symbol, pos in list(self.portfolio.positions.items()):
            tick = MT5PriceFeed.get_tick(symbol)
            if tick is None:
                logger.warning(f"[Monitor] Cannot fetch price for {symbol}.")
                continue

            current_price: float = tick["mid"]
            self.portfolio.update_price(symbol, current_price)

            if pos.entry_price <= 0:
                continue

            if pos.direction == "BUY":
                profit = current_price / pos.entry_price - 1.0
            else:  # SELL（空头）
                profit = pos.entry_price / current_price - 1.0

            # ── 止损（所有模式）────────────────────────────────────
            if profit < Config.STOP_LOSS_PCT:
                logger.warning(
                    f"[Monitor] STOP LOSS: {symbol} {pos.direction} "
                    f"profit={profit:.2%}"
                )
                ok = self.trader.close_position(
                    symbol, pos.lot_size, pos.direction, pos.ticket
                )
                if ok:
                    self.portfolio.close_position(symbol)
                continue

            # hybrid 模式只做止损，跳过下面的止盈/追踪
            if Config.EXIT_MODE == "hybrid":
                continue

            # ── 部分止盈（risk 模式）────────────────────────────────
            if profit > Config.TAKE_PROFIT_PCT and not pos.is_partial_closed:
                half = round(pos.lot_size / 2, 2)
                if half > 0:
                    logger.info(f"[Monitor] Partial TP: {symbol} profit={profit:.2%}")
                    ok = self.trader.close_position(
                        symbol, half, pos.direction, pos.ticket
                    )
                    if ok:
                        pos.is_partial_closed = True
                        self.portfolio.save_state()
                continue

            # ── 追踪止损（risk 模式，多头用最高价，空头用最低价）──
            if profit > Config.TRAILING_ACTIVATION:
                if pos.direction == "BUY" and pos.highest_price > 0:
                    drawdown = (pos.highest_price - current_price) / pos.highest_price
                    if drawdown > Config.TRAILING_DROP:
                        logger.warning(
                            f"[Monitor] TRAILING STOP (long): {symbol} "
                            f"dd={drawdown:.2%}"
                        )
                        ok = self.trader.close_position(
                            symbol, pos.lot_size, pos.direction, pos.ticket
                        )
                        if ok:
                            self.portfolio.close_position(symbol)
                elif pos.direction == "SELL" and pos.lowest_price > 0:
                    # 空头：从最低价反弹超过 TRAILING_DROP 则止损
                    rebound = (current_price - pos.lowest_price) / pos.lowest_price
                    if rebound > Config.TRAILING_DROP:
                        logger.warning(
                            f"[Monitor] TRAILING STOP (short): {symbol} "
                            f"rebound={rebound:.2%}"
                        )
                        ok = self.trader.close_position(
                            symbol, pos.lot_size, pos.direction, pos.ticket
                        )
                        if ok:
                            self.portfolio.close_position(symbol)

    # ──────────────────────────────────────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────────────────────────────────────

    def _calc_lot(self, symbol: str) -> float:
        """计算手数（复用 MT5RiskEngine）。"""
        account = self.trader.get_account_info()
        if account is None:
            return 0.0
        equity    = account["equity"]
        stop_pips = 20.0   # 通用保守估计；可由品种规格动态获取
        return self.risk.calculate_lot(symbol, equity, stop_pips)

    def _get_price(self, symbol: str) -> float:
        """获取当前中间价，失败返回 0.0。"""
        tick = MT5PriceFeed.get_tick(symbol)
        return tick["mid"] if tick else 0.0
