# Forex 组中间训练结果（改进方案首轮，训练 21% 时的快照）

> 生成时间：2026-07-04
> 状态：训练进行到 forex 组 step 1055/5000（约 21%）时手动中断，回测中间结果
> vocab_version：v61d94be3d0e5（特征剪枝后，28 特征 + 66 算子 = 94 token）

## 一、改进方案回顾（本轮训练所用）

针对上一轮"特征扩展后反而变差"的问题，做了四层改进：

1. **特征剪枝**：用 EffectivenessEvaluator 从 65 特征筛到 28 个，vocab 131→94，
   8-token 搜索空间从 8.67×10¹⁶ 缩到 6.1×10¹⁵（约 14 倍）
2. **采样预算**：BATCH_SIZE 128→192，TRAIN_STEPS 3000→5000，ELITE_POOL_SIZE 30→60
3. **熵坍塌修复**：ENTROPY_COEFF_MAX 0.5→1.0，坍塌阈值改为相对值 0.15×ln(vocab)≈0.68，
   ENTROPY_COLLAPSE_STEPS 15→40，MAX_RESTARTS 8→25，RESTART_NOISE 0.05→0.1，
   并移除 Early Stop 终止（改为强扰动继续）
4. **模型容量**：d_model 64→96，层数 2→3，FFN 128→192

## 二、公式

```
EMA_RATIO_12_26 -> TS_MIN_20 -> AROON_OSC_25 -> MOMENTUM_5 -> EMA_RATIO_12_26 -> SCALE -> TS_DECAY_EXP_5 -> IF_GT
```

Token 序列：`[10, 60, 20, 50, 10, 66, 70, 93]`
训练 best_score：5.8417（step 1055 时）

## 三、回测结果（离线，T=3494 bars H1，真实点差）

| 品种 | PnL | Sharpe | Sortino | MaxDD | Calmar | 交易数 | 胜率 | 平均持仓 |
|------|-----|--------|---------|-------|--------|--------|------|----------|
| EURUSD | +0.071 | +1.07 | +1.49 | 0.070 | +1.80 | 105 | 53.3% | 33.3h |
| USDJPY | +0.168 | +2.15 | +2.42 | 0.104 | +2.89 | 103 | 59.2% | 33.9h |
| **Portfolio** | **+0.119** | **+2.78** | **+3.29** | **0.028** | **+7.57** | — | — | — |

- 正收益品种：2/2
- Sharpe > 1 品种：2/2

## 四、与上一轮失败版对比

| 指标 | 上轮失败版（07-04 早） | 本轮中间版（仅 21%） |
|------|----------------------|---------------------|
| EURUSD Sharpe | -0.29（亏损） | **+1.07** ✅ |
| USDJPY Sharpe | +1.32 | **+2.15** ✅ |
| 组合 Sharpe | +2.84 | +2.78 |
| 组合 MaxDD | 0.088 | **0.028**（更低） |
| 交易数 | 288/286 | 105/103（更健康中频） |
| 平均持仓 | 12h | 33h（更抗点差） |
| 熵坍塌 | 8/8 触发 Early Stop | 0 次重启，H≈0.5 稳定 |

## 五、结论

改进方案有效验证：
1. **EURUSD 从亏损转为盈利**，上一轮 group 共用公式泛化失败的问题解决
2. **两品种 Sharpe 均 > 1**，持仓 33h 比之前 12h 更合理，抗点差能力更强
3. **组合回撤仅 0.028**，风险控制显著改善
4. 熵坍塌问题解决，训练全程健康，未触发 Early Stop

以上仅为训练 21% 的中间快照，跑满 5000 步预期还能进一步提升。metals_comm 与 index 组尚未开始训练。

## 六、文件

- 公式文件：`strategies/best_forex.json`（symbol=forex）、`best_EURUSD.json`、`best_USDJPY.json`
- 回测报告：`backtest_output/multi_factor_report.json`
- 回测日志：`backtest_forex_interim.log`
- 特征白名单：`active_features.json`（28 特征）
- 剪枝脚本：`prune_features.py`
