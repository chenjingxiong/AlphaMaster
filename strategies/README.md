# AlphaGPT 策略库

训练出的有效因子公式，按版本存档。每个文件包含：公式 tokens、回测指标、训练配置、分析结论。

## 使用方式

```python
import json
with open("strategies/strategy_v1_20260702.json") as f:
    s = json.load(f)
tokens = s["formula"]["tokens"]
```

注意：`vocab_version` 必须与当前 `model_core/vocab.py` 中的 `VOCAB_VERSION` 一致，否则 token 语义不同。

---

## 策略列表

| ID | 日期 | 核心因子 | 组合PnL | 正收益品种 | Sortino最强 | 状态 |
|----|------|--------|--------|---------|-----------|------|
| [v1](strategy_v1_20260702.json) | 2026-07-02 | `TS_RANK_10(MA_DIFF)` 均线差排名 | **+118.9%** | 3/5 (USTECm/XAUUSDm/US500m) | USTECm +2.50 | ✅ 存档 |

---

## 版本说明

### v1 — MA_RANK_VOL_TREND（2026-07-02）

**有效核心**：`TS_RANK_10(MA_DIFF)`
- MA_DIFF = MA10/MA30 - 1（短长均线差）
- TS_RANK_10 = 10期历史排名（值域 [0,1)）
- 公式完整形式带 GATE/RVOL/SLOPE20 的冗余结构，实际只有核心部分在工作

**适用品种**：风险资产（指数 > 黄金 > 外汇）

**不适用**：EUR/USD、USD/JPY 趋势信号噪声过大

**训练条件**
- 阶段 A，8-token 公式，300步（实际跑 620步，冠军在 step 125 出现）
- 5 品种，H1 周期，约 1.8 年历史数据
- vocab v2.0（20个特征）

**下一步**：切换阶段B（14-token，500步）搜索更复杂结构
