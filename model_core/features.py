"""
model_core/features.py -- MT5 Feature Engineering (20 features)

Features:
  Trend (0-4):   RET, RET5, RET20, MA_DIFF, SLOPE20
  Volatility (5-8): ATR, RVOL, HL_RANGE, VOL_REGIME
  Reversal (9-13):  DEV, DEV60, RSI14, PRESSURE, AC1
  Volume (14-16):   VOL_RATIO, VOL_Z, PV_CORR
  Cross-asset (17-19): REL_RET5, REL_RET20, REL_VOL

Output: [N, 20, T], all normalized, no NaN/Inf.
"""
import torch


class MT5FeatureEngineer:
    """MT5 Feature Engineer (20 features)."""

    INPUT_DIM    = 20
    _CLIP_BOUND  = 5.0
    _EPS         = 1e-9
    _MA_WINDOW   = 20

    # ── rolling helpers ──────────────────────────────────────────────────

    @staticmethod
    def _rolling_mean(x: torch.Tensor, w: int) -> torch.Tensor:
        N, T = x.shape
        pad  = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x], dim=1).unfold(1, w, 1).mean(dim=-1)

    @classmethod
    def _ma(cls, x: torch.Tensor, w: int) -> torch.Tensor:
        return cls._rolling_mean(x, w)

    @classmethod
    def _ma20(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._rolling_mean(x, cls._MA_WINDOW)

    @staticmethod
    def _rolling_std(x: torch.Tensor, w: int) -> torch.Tensor:
        N, T = x.shape
        pad  = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        wnd  = torch.cat([pad, x], dim=1).unfold(1, w, 1)
        m    = wnd.mean(dim=-1, keepdim=True)
        return ((wnd - m) ** 2).mean(dim=-1).sqrt() + 1e-9

    @staticmethod
    def _atr(close: torch.Tensor, high: torch.Tensor,
             low: torch.Tensor, w: int = 14) -> torch.Tensor:
        pc = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        tr = torch.stack([high - low,
                          (high - pc).abs(),
                          (low  - pc).abs()], dim=-1).max(dim=-1).values
        return MT5FeatureEngineer._rolling_mean(tr, w)

    @staticmethod
    def _rvol(close: torch.Tensor, w: int = 20) -> torch.Tensor:
        eps = MT5FeatureEngineer._EPS
        ret = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = torch.cat([torch.zeros_like(close[:, :1]), ret], dim=1)
        N   = ret.shape[0]
        pad = torch.zeros(N, w - 1, device=ret.device, dtype=ret.dtype)
        wnd = torch.cat([pad, ret], dim=1).unfold(1, w, 1)
        m   = wnd.mean(dim=-1, keepdim=True)
        return ((wnd - m) ** 2).mean(dim=-1).sqrt() + 1e-9

    @staticmethod
    def _ac1(close: torch.Tensor, w: int = 20) -> torch.Tensor:
        eps = MT5FeatureEngineer._EPS
        ret = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = torch.cat([torch.zeros_like(close[:, :1]), ret], dim=1)
        N   = ret.shape[0]
        pad = torch.zeros(N, w, device=ret.device, dtype=ret.dtype)
        wnd = torch.cat([pad, ret], dim=1).unfold(1, w + 1, 1)
        x, y = wnd[:, :, :-1], wnd[:, :, 1:]
        xm, ym = x.mean(dim=-1, keepdim=True), y.mean(dim=-1, keepdim=True)
        cov = ((x - xm) * (y - ym)).mean(dim=-1)
        sx  = ((x - xm) ** 2).mean(dim=-1).sqrt()
        sy  = ((y - ym) ** 2).mean(dim=-1).sqrt()
        out = cov / (sx * sy + 1e-8)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _linear_slope(x: torch.Tensor, w: int) -> torch.Tensor:
        """Causal linear regression slope, normalized by price level."""
        N, T  = x.shape
        eps   = MT5FeatureEngineer._EPS
        pad   = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        wnd   = torch.cat([pad, x], dim=1).unfold(1, w, 1)   # [N, T, w]
        tidx  = torch.arange(w, dtype=x.dtype, device=x.device)
        tc    = tidx - tidx.mean()
        tvar  = (tc ** 2).sum()
        xm    = wnd.mean(dim=-1, keepdim=True)
        slope = ((wnd - xm) * tc).sum(dim=-1) / (tvar + eps)
        slope = slope / (xm.squeeze(-1) + eps)
        return torch.nan_to_num(slope, nan=0.0)

    @staticmethod
    def _rsi(close: torch.Tensor, w: int = 14) -> torch.Tensor:
        """RSI normalized to [-1, 1]."""
        diff   = close - torch.cat([close[:, :1], close[:, :-1]], dim=1)
        gains  = torch.relu(diff)
        losses = torch.relu(-diff)
        avg_g  = MT5FeatureEngineer._rolling_mean(gains,  w)
        avg_l  = MT5FeatureEngineer._rolling_mean(losses, w)
        rs     = (avg_g + 1e-9) / (avg_l + 1e-9)
        rsi    = 100.0 - (100.0 / (1.0 + rs))
        return (rsi - 50.0) / 50.0

    @staticmethod
    def _ts_corr(x: torch.Tensor, y: torch.Tensor, w: int) -> torch.Tensor:
        """Causal sliding Pearson correlation."""
        N, T  = x.shape
        px    = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        py    = torch.zeros(N, w - 1, dtype=y.dtype, device=y.device)
        wx    = torch.cat([px, x], dim=1).unfold(1, w, 1)
        wy    = torch.cat([py, y], dim=1).unfold(1, w, 1)
        mx, my = wx.mean(dim=-1, keepdim=True), wy.mean(dim=-1, keepdim=True)
        cov   = ((wx - mx) * (wy - my)).mean(dim=-1)
        sx    = ((wx - mx) ** 2).mean(dim=-1).sqrt()
        sy    = ((wy - my) ** 2).mean(dim=-1).sqrt()
        mask  = (sx < 1e-6) | (sy < 1e-6)
        corr  = cov / (sx * sy + 1e-8)
        corr[mask] = 0.0
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _robust_norm(x: torch.Tensor) -> torch.Tensor:
        med = x.median(dim=1, keepdim=True).values
        mad = (x - med).abs().median(dim=1, keepdim=True).values + 1e-6
        return torch.clamp((x - med) / mad,
                           -MT5FeatureEngineer._CLIP_BOUND,
                            MT5FeatureEngineer._CLIP_BOUND)

    @staticmethod
    def _clean(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _ret20(close: torch.Tensor) -> torch.Tensor:
        """兼容方法：20期对数动量，前20位补0，返回 [N, T]。

        保留此 helper 以兼容外部调用和测试代码。
        compute_features() 内部等效计算见 ret20_raw。
        """
        N, T = close.shape
        eps  = MT5FeatureEngineer._EPS
        raw  = torch.log(close[:, 20:] / (close[:, :-20] + eps))
        pad  = torch.zeros(N, 20, device=close.device, dtype=close.dtype)
        return torch.cat([pad, raw], dim=1)

    # ── main ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_features(raw_dict: dict) -> torch.Tensor:
        """Compute 20 features, returns [N, 20, T]."""
        close  = raw_dict["close"].float()
        open_  = raw_dict["open"].float()
        high   = raw_dict["high"].float()
        low    = raw_dict["low"].float()
        volume = raw_dict["volume"].float()

        N, T = close.shape
        eps  = MT5FeatureEngineer._EPS
        fe   = MT5FeatureEngineer

        def norm(x):
            return fe._clean(fe._robust_norm(fe._clean(x)))

        # ── Trend (0-4) ──────────────────────────────────────────────────
        # 0: RET
        ret_raw = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = norm(torch.cat([torch.zeros(N, 1, device=close.device), ret_raw], dim=1))

        # 1: RET5
        ret5_raw = torch.log(close[:, 5:] / (close[:, :-5] + eps))
        ret5 = norm(torch.cat([torch.zeros(N, 5, device=close.device), ret5_raw], dim=1))

        # 2: RET20
        ret20_raw = torch.log(close[:, 20:] / (close[:, :-20] + eps))
        ret20 = norm(torch.cat([torch.zeros(N, 20, device=close.device), ret20_raw], dim=1))

        # 3: MA_DIFF (MA10/MA30 - 1)
        ma_diff = norm(fe._ma(close, 10) / (fe._ma(close, 30) + eps) - 1.0)

        # 4: SLOPE20
        slope = norm(fe._linear_slope(close, 20))

        # ── Volatility (5-8) ─────────────────────────────────────────────
        # 5: ATR
        atr_raw = fe._atr(close, high, low)
        atr = norm(torch.log1p(fe._clean(atr_raw.clamp(min=0))))

        # 6: RVOL
        rvol_raw = fe._rvol(close)
        rvol = norm(torch.log1p(fe._clean(rvol_raw.clamp(min=0))))

        # 7: HL_RANGE
        hl_range = norm((high - low) / (close + eps))

        # 8: VOL_REGIME (ATR / MA20(ATR) - 1)
        ma_atr = fe._ma(atr_raw, 20)
        vol_regime = norm(atr_raw / (ma_atr + eps) - 1.0)

        # ── Reversal (9-13) ──────────────────────────────────────────────
        # 9: DEV (price deviation from MA20)
        ma20c = fe._ma20(close)
        dev = norm((close - ma20c) / (ma20c + eps))

        # 10: DEV60
        ma60 = fe._ma(close, 60)
        dev60 = norm((close - ma60) / (ma60 + eps))

        # 11: RSI14
        rsi = fe._clean(torch.clamp(fe._rsi(close, 14), -1.0, 1.0))

        # 12: PRESSURE
        pressure = fe._clean(torch.clamp((close - open_) / (high - low + eps), -1.0, 1.0))

        # 13: AC1
        ac1 = fe._clean(torch.clamp(fe._ac1(close), -1.0, 1.0))

        # ── Volume (14-16) ───────────────────────────────────────────────
        # 14: VOL_RATIO
        ma20v = fe._ma20(volume)
        vol_ratio = norm(volume / (ma20v + eps))

        # 15: VOL_Z
        std20v = fe._rolling_std(volume, 20)
        vol_z  = fe._clean(torch.clamp((volume - ma20v) / (std20v + eps), -5.0, 5.0))

        # 16: PV_CORR (price-volume 10-bar correlation)
        log_vol_ratio = torch.log1p(fe._clean(vol_ratio.clamp(min=-0.99)))
        pv_corr = fe._clean(torch.clamp(fe._ts_corr(ret, log_vol_ratio, 10), -1.0, 1.0))

        # ── Cross-asset relative strength (17-19) ────────────────────────
        # 17: REL_RET5
        rel_ret5 = norm(ret5 - ret5.mean(dim=0, keepdim=True))

        # 18: REL_RET20
        rel_ret20 = norm(ret20 - ret20.mean(dim=0, keepdim=True))

        # 19: REL_VOL
        rel_vol = norm(rvol - rvol.mean(dim=0, keepdim=True))

        features = torch.stack([
            ret, ret5, ret20, ma_diff, slope,       # trend  0-4
            atr, rvol, hl_range, vol_regime,         # vol    5-8
            dev, dev60, rsi, pressure, ac1,          # rev    9-13
            vol_ratio, vol_z, pv_corr,               # volume 14-16
            rel_ret5, rel_ret20, rel_vol,            # cross  17-19
        ], dim=1)
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
