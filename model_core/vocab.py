from dataclasses import dataclass

from .ops import OPS_CONFIG

# 词表版本：特征顺序变化时递增，旧 checkpoint / best_strategy.json 与新版本不兼容
VOCAB_VERSION = "2.0"   # v1.0=10特征, v2.0=20特征(2026-07)


FEATURE_NAMES = (
    # 趋势类（0-4）
    "RET",          # 0  单期对数收益率
    "RET5",         # 1  5期动量
    "RET20",        # 2  20期动量
    "MA_DIFF",      # 3  MA10/MA30-1 短长均线差
    "SLOPE20",      # 4  20期线性回归斜率
    # 波动类（5-8）
    "ATR",          # 5  平均真实波幅
    "RVOL",         # 6  已实现波动率
    "HL_RANGE",     # 7  (high-low)/close
    "VOL_REGIME",   # 8  ATR/MA(ATR)-1 波动率状态
    # 反转类（9-13）
    "DEV",          # 9  (close-MA20)/MA20
    "DEV60",        # 10 (close-MA60)/MA60
    "RSI14",        # 11 RSI归一化[-1,1]
    "PRESSURE",     # 12 (close-open)/(high-low)
    "AC1",          # 13 一阶自相关
    # 成交量类（14-16）
    "VOL_RATIO",    # 14 volume/MA20(volume)
    "VOL_Z",        # 15 量能z分数
    "PV_CORR",      # 16 价量10期相关系数
    # 跨资产相对强弱（17-19）
    "REL_RET5",     # 17 相对5期收益（截面去均值）
    "REL_RET20",    # 18 相对20期收益（截面去均值）
    "REL_VOL",      # 19 相对波动率（截面去均值）
)


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
