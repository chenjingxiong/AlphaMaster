import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """降低稀疏度：阈值从 3σ 改为 1.5σ，让更多时间步有非零输出"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.tanh(z - 1.5)   # tanh 软化，不再产生全零区间

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

@torch.jit.script
def _op_wma(x: torch.Tensor) -> torch.Tensor:
    """加权移动平均（权重 3,2,1），平滑信号，减少剥头皮"""
    return (3.0 * x + 2.0 * _ts_delay(x, 1) + 1.0 * _ts_delay(x, 2)) / 6.0

OPS_CONFIG = [
    ('ADD',    lambda x, y: x + y,        2),
    ('SUB',    lambda x, y: x - y,        2),
    ('MUL',    lambda x, y: x * y,        2),
    ('DIV',    lambda x, y: x / (y + 1e-6), 2),
    ('NEG',    lambda x: -x,              1),
    ('ABS',    torch.abs,                  1),
    ('SIGN',   torch.sign,                 1),
    ('GATE',   _op_gate,                   3),
    ('JUMP',   _op_jump,                   1),   # 已降低稀疏度
    ('DECAY',  _op_decay,                  1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3',   lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1),
]

# ── 时序滑动窗口辅助函数（不使用 @torch.jit.script，lambda 不兼容 JIT）──────

def _ts_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    """unfold 实现因果滑动窗口，返回 [N, T, d] 的窗口张量。"""
    N, T = x.shape
    pad = torch.zeros(N, d - 1, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1).unfold(1, d, 1)  # [N, T, d]


def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动均值，返回 [N, T]。"""
    return _ts_rolling(x, d).mean(dim=-1)


def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动标准差（ddof=0），返回 [N, T]，下界 1e-6。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    m = w.mean(dim=-1, keepdim=True)
    std = ((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6
    return torch.nan_to_num(std, nan=0.0)


def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动排名（严格小于当前值的比例），返回 [N, T]，值域 [0, 1)。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    cur = w[:, :, -1:]                             # 当前值，[N, T, 1]
    rank = (w < cur).float().mean(dim=-1)          # [N, T]
    return torch.nan_to_num(rank, nan=0.0)


def _ts_corr_10(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x 与 y 的 10 周期因果滑动 Pearson 相关系数，返回 [N, T]，值域 [-1, 1]。
    当 x 或 y 在窗口内为常数（std < 1e-6）时，该位置输出 0。
    """
    d = 10
    wx = _ts_rolling(x, d)                         # [N, T, 10]
    wy = _ts_rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    sx = ((wx - mx) ** 2).mean(dim=-1).sqrt()      # [N, T]
    sy = ((wy - my) ** 2).mean(dim=-1).sqrt()
    # 常数窗口（std < 1e-6）输出 0
    mask = (sx < 1e-6) | (sy < 1e-6)
    corr = cov / (sx * sy + 1e-8)
    corr[mask] = 0.0
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


# ── 追加时序算子到 OPS_CONFIG（token id 从 feat_offset+12 开始）────────────

OPS_CONFIG += [
    ('TS_MEAN_5',  lambda x: _ts_mean(x, 5),  1),
    ('TS_MEAN_10', lambda x: _ts_mean(x, 10), 1),
    ('TS_MEAN_20', lambda x: _ts_mean(x, 20), 1),
    ('TS_STD_5',   lambda x: _ts_std(x, 5),   1),
    ('TS_STD_10',  lambda x: _ts_std(x, 10),  1),
    ('TS_STD_20',  lambda x: _ts_std(x, 20),  1),
    ('TS_RANK_5',  lambda x: _ts_rank(x, 5),  1),
    ('TS_RANK_10', lambda x: _ts_rank(x, 10), 1),
    ('TS_RANK_20', lambda x: _ts_rank(x, 20), 1),
    ('TS_CORR_10', _ts_corr_10,                2),
    # ── 趋势 / 动量类算子（新增，token id = feat_offset+22~27）────────────
    # MOMENTUM_5: 短期均线 - 长期均线，捕捉趋势方向
    ('MOMENTUM_5',  lambda x: _ts_mean(x, 5)  - _ts_mean(x, 20), 1),
    # MOMENTUM_10: 中期动量
    ('MOMENTUM_10', lambda x: _ts_mean(x, 10) - _ts_mean(x, 20), 1),
    # TS_MAX_10: 10周期最大值，捕捉强势突破
    ('TS_MAX_10',   lambda x: _ts_rolling(x, 10).max(dim=-1).values, 1),
    # TS_MIN_10: 10周期最小值，捕捉弱势突破
    ('TS_MIN_10',   lambda x: _ts_rolling(x, 10).min(dim=-1).values, 1),
    # WMA: 加权移动平均，平滑信号
    ('WMA',         _op_wma,  1),
    # DELAY4: 延迟4根bar，构建中期动量差
    ('DELAY4',      lambda x: _ts_delay(x, 4), 1),
]

assert len(OPS_CONFIG) == 28, f"OPS_CONFIG 长度应为 28，实际为 {len(OPS_CONFIG)}"
