"""东方财富数据源（A 股 / 指数，直连 push2his HTTP API，无需第三方库）。"""
from __future__ import annotations

from datetime import datetime, timezone

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

# 项目周期 -> 东财 klt
_KLT = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "1d": "101",
    "1w": "102",
    "1M": "103",
}

_PRESETS = ["600519", "000001", "300750", "601318", "000858", "sh000300", "sz399006"]

_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def _secid(symbol: str) -> str:
    """把 A 股代码转成东财 secid（1.=上交所, 0.=深交所）。"""
    s = symbol.strip().lower()
    if s.startswith("sh"):
        return f"1.{s[2:]}"
    if s.startswith("sz"):
        return f"0.{s[2:]}"
    code = s
    if not code.isdigit() or len(code) != 6:
        raise DataSourceUnavailable("A 股代码需为 6 位数字，或 sh/sz 前缀的指数代码")
    # 6/5/9 开头（含指数 000xxx 需前缀 sh）→ 上交所；0/3 → 深交所
    if code[0] in ("6", "5", "9") or code.startswith("11") or code.startswith("13"):
        return f"1.{code}"
    return f"0.{code}"


def _parse_dt(s: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return 0


class EastMoneySource(DataSource):
    kind = "eastmoney"
    label = "东方财富"

    def available(self) -> tuple[bool, str]:
        try:
            import requests  # noqa: F401
        except ImportError:
            return (False, "未安装 requests：pip install requests")
        return (True, "A 股 / 指数 · 6 位代码或 sh/sz 指数")

    def supported_timeframes(self) -> list[str]:
        return list(_KLT.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _KLT:
            raise DataSourceUnavailable(f"东方财富不支持周期 {timeframe}")
        try:
            import requests
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 requests") from exc

        secid = _secid(symbol)
        lmt = n + 2 if drop_forming else n + 1
        params = {
            "secid": secid,
            "klt": _KLT[timeframe],
            "fqt": "1",  # 前复权
            "fields1": "f1,f2,f3,f4,f5",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "beg": "0",
            "end": "20500101",
            "lmt": str(lmt),
        }
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        try:
            resp = requests.get(_KLINE_URL, params=params, headers=headers, timeout=8)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise DataSourceUnavailable(f"东方财富拉取失败: {exc}") from exc

        klines = (payload.get("data") or {}).get("klines") or []
        if not klines:
            raise DataSourceUnavailable(f"东方财富无数据：{symbol}")

        bars: list[Bar] = []
        for line in klines:
            # date,open,close,high,low,volume,amount
            parts = line.split(",")
            if len(parts) < 6:
                continue
            bars.append(
                Bar(
                    ts=_parse_dt(parts[0]),
                    open=float(parts[1]),
                    close=float(parts[2]),
                    high=float(parts[3]),
                    low=float(parts[4]),
                    volume=float(parts[5]),
                )
            )
        # 已升序；剔除最后一根（盘中为正在形成的 bar）
        if drop_forming and len(bars) > 1:
            bars = bars[:-1]
        return bars[-n:]
