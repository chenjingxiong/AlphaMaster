"""通达信数据源（pytdx，免费行情服务器，A 股 / 指数）。"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

# 通达信行情服务器（多个备选）
_SERVERS = [
    ("115.238.90.165", 7709),
    ("180.153.18.170", 7709),
    ("119.147.212.81", 7709),
    ("14.17.75.71", 7709),
    ("59.173.18.77", 7709),
]

# 项目周期 -> pytdx category
_CAT = {
    "1m": 8,
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "1h": 3,
    "1d": 9,
    "1w": 5,
    "1M": 6,
}

_PRESETS = ["600519", "000001", "300750", "601318", "000858", "sh000001", "sz399006"]


def _parse_market(code: str) -> tuple[int, str]:
    """返回 (market, pure_code)。1=上海, 0=深圳。"""
    c = code.strip().upper()
    if c.startswith("SH"):
        return 1, c[2:]
    if c.startswith("SZ"):
        return 0, c[2:]
    if c[:1] in ("6", "5", "9") or c.startswith("11") or c.startswith("13"):
        return 1, c
    return 0, c


def _parse_dt(s: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(str(s), fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return 0


class TongdaxinSource(DataSource):
    kind = "tongdaxin"
    label = "通达信"

    def __init__(self) -> None:
        self._api = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            import pytdx  # noqa: F401
        except ImportError:
            return (False, "未安装 pytdx：pip install pytdx")
        return (True, "免费行情服务器 · A 股 / 指数")

    def supported_timeframes(self) -> list[str]:
        return list(_CAT.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def connect(self) -> None:
        if self._api is not None:
            return
        try:
            from pytdx.hq import TdxHq_API
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 pytdx") from exc
        api = TdxHq_API()
        for host, port in _SERVERS:
            try:
                if api.connect(host, port):
                    self._api = api
                    return
            except Exception:
                continue
        raise DataSourceUnavailable("通达信所有行情服务器连接失败")

    def disconnect(self) -> None:
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
        self._api = None

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _CAT:
            raise DataSourceUnavailable(f"通达信不支持周期 {timeframe}")
        market, code = _parse_market(symbol)
        cat = _CAT[timeframe]
        want = min(max(n + 2, 20), 800)  # 单次上限 800

        with self._lock:
            self.connect()
            try:
                raw = self._api.get_security_bars(cat, market, code, 0, want)
            except Exception as exc:
                # 连接可能失效，重连一次
                self._api = None
                self.connect()
                raw = self._api.get_security_bars(cat, market, code, 0, want)

        if not raw:
            raise DataSourceUnavailable(f"通达信无数据：{symbol}")

        bars: list[Bar] = []
        for r in raw:
            bars.append(
                Bar(
                    ts=_parse_dt(r.get("datetime", "")),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r.get("vol", 0.0) or 0.0),
                )
            )
        bars.sort(key=lambda b: b.ts)  # 保证升序
        if drop_forming and len(bars) > 1:
            bars = bars[:-1]
        return bars[-n:]
