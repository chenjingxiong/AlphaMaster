"""文件选择（桌面 tkinter / 无头容器服务端扫描双模式）。

桌面模式（有 DISPLAY 且 tkinter 可用）：弹出原生文件选择对话框。
无头模式（Docker/服务器，无 DISPLAY 或缺 tkinter）：扫描数据/策略目录，
返回最近修改的文件，让 Web UI 流程不依赖图形环境。

无头模式下建议改用前端的「文件列表 + 上传」流程（/api/data-file/list、
/api/data-file/upload），这里的 pick_* 仅为兼容旧调用方而保留自动选择行为。
"""
from __future__ import annotations

import os
from pathlib import Path


# 无头环境下的扫描根目录：容器内是 /app/data 与 /app/strategies；
# 桌面开发环境回退到项目根下的 data/、strategies/。
_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIRS = [
    Path(os.getenv("KLINE_CACHE_DIR", "")) if os.getenv("KLINE_CACHE_DIR") else None,
    _ROOT / "data",
    _ROOT / "data" / "kline_cache",
]
_STRATEGY_DIRS = [
    _ROOT / "strategies",
]


def _is_headless() -> bool:
    """无图形环境（无 DISPLAY 且非 Windows/macOS GUI）或 tkinter 不可用 → True。"""
    if os.name == "nt":
        return False  # Windows 桌面
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return True
    # 有 DISPLAY 也要确认 tkinter 真能 import（Linux 服务器常装了 Tcl/Tk 库但不全）
    try:
        import tkinter  # noqa: F401
        return False
    except Exception:
        return True


def _native_pick(title: str, filetypes: list[tuple[str, str]]) -> str | None:
    """桌面模式：弹原生选择框。"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path or None


def _scan_latest(dirs: list[Path], suffix: str, predicate=None) -> str | None:
    """无头模式：在给定目录递归扫描，返回最近修改的匹配文件路径。"""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for base in dirs:
        if not base:
            continue
        try:
            base = base.resolve()
        except Exception:
            continue
        if not base.exists():
            continue
        for p in base.rglob(f"*{suffix}"):
            if p in seen:
                continue
            seen.add(p)
            if predicate and not predicate(p):
                continue
            candidates.append(p)
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def pick_parquet_file() -> str | None:
    """选择 K 线 Parquet 文件。

    桌面模式弹原生对话框；无头模式返回 /app/data 下最近修改的 .parquet。
    """
    if _is_headless():
        return _scan_latest(_DATA_DIRS, ".parquet")
    try:
        return _native_pick(
            "选择 K 线 Parquet 文件",
            [("Parquet K线", "*.parquet"), ("所有文件", "*.*")],
        )
    except Exception:
        # tkinter 加载失败时回退到扫描
        return _scan_latest(_DATA_DIRS, ".parquet")


def pick_strategy_file() -> str | None:
    """选择策略 JSON 文件。

    桌面模式弹原生对话框；无头模式返回 strategies/ 下最近修改的 .json。
    """
    if _is_headless():
        return _scan_latest(_STRATEGY_DIRS, ".json")
    try:
        return _native_pick(
            "选择策略 JSON 文件",
            [("策略 JSON", "*.json"), ("所有文件", "*.*")],
        )
    except Exception:
        return _scan_latest(_STRATEGY_DIRS, ".json")


def list_parquet_files() -> list[dict]:
    """列出所有可训练 Parquet 文件（无头/桌面通用），按修改时间倒序。"""
    seen: set[Path] = set()
    items: list[Path] = []
    for base in _DATA_DIRS:
        if not base:
            continue
        try:
            base = base.resolve()
        except Exception:
            continue
        if not base.exists():
            continue
        for p in base.rglob("*.parquet"):
            if p not in seen:
                seen.add(p)
                items.append(p)
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [_file_info(p) for p in items]


def list_strategy_files() -> list[dict]:
    """列出所有策略 JSON 文件，按修改时间倒序。"""
    items: list[Path] = []
    for base in _STRATEGY_DIRS:
        if not base.exists():
            continue
        items.extend(p for p in base.rglob("*.json"))
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "path": str(p.resolve()),
            "filename": p.name,
            "size": p.stat().st_size,
            "mtime": p.stat().st_mtime,
        }
        for p in items
    ]


def _file_info(p: Path) -> dict:
    """单个 Parquet 文件的展示元数据（尝试从文件名解析 symbol/timeframe）。"""
    info = {
        "path": str(p.resolve()),
        "filename": p.name,
        "size": p.stat().st_size,
        "mtime": p.stat().st_mtime,
        "symbol": None,
        "timeframe": None,
    }
    try:
        from data_pipeline.parquet_manager import parse_parquet_filename

        symbol, timeframe = parse_parquet_filename(p)
        info["symbol"] = symbol
        info["timeframe"] = timeframe
    except Exception:
        pass
    return info
