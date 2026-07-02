"""analyze_training.py — 训练结果快速分析"""
import sys, json, torch
sys.path.insert(0, '.')
from model_core.vocab import FORMULA_VOCAB
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from model_core.vm import StackVM

with open('best_mt5_strategy.json') as f:
    tokens = json.load(f)
with open('training_history.json') as f:
    h = json.load(f)

names  = FORMULA_VOCAB.token_names
n      = len(h['step'])
vals   = h['val_score']
bests  = h['best_score']
rews   = h['avg_reward']
ents   = h['entropy']
ics    = h['ic_mean']
sorts  = h.get('sortino', [0]*n)

formula_str = ' -> '.join(names[t] for t in tokens)
print(f'=== 训练完成 总步数={n} ===')
print(f'全局最优 BestScore : {max(bests):.4f}')
print(f'末步批次 ValScore  : {vals[-1]:+.4f}')
print(f'末步 AvgReward     : {rews[-1]:+.4f}')
print(f'末步 Entropy       : {ents[-1]:.3f}')
print(f'末步 IC            : {ics[-1]:+.5f}')
print(f'末步 Sortino(batch): {sorts[-1]:+.4f}')
print(f'最优公式: {formula_str}')
print()

print('=== BestScore 演进 ===')
prev = bests[0]
print(f'  step  0: {prev:.4f}')
for i in range(1, n):
    if bests[i] > prev + 0.05:
        print(f'  step{h["step"][i]:4d}: {prev:.4f} -> {bests[i]:.4f}  (+{bests[i]-prev:.4f})')
        prev = bests[i]

print()
print('=== 分阶段统计 ===')
for s, e in [(0,100),(100,200),(200,300),(300,400),(400,500)]:
    sv  = [vals[i]  for i in range(n) if s <= h['step'][i] < e]
    sr  = [rews[i]  for i in range(n) if s <= h['step'][i] < e]
    se  = [ents[i]  for i in range(n) if s <= h['step'][i] < e]
    sic = [ics[i]   for i in range(n) if s <= h['step'][i] < e]
    if sv:
        print(f'  {s:3d}-{e}: val_mean={sum(sv)/len(sv):+.3f}'
              f'  val_max={max(sv):+.3f}'
              f'  rew={sum(sr)/len(sr):+.3f}'
              f'  H={sum(se)/len(se):.2f}'
              f'  IC={sum(sic)/len(sic):+.5f}')

# val 转正
for i, v in enumerate(vals):
    if v > 0:
        print(f'\n  val_score 首次转正: step {h["step"][i]} (val={v:+.4f})')
        break

print()
print('=== 最优公式交易频率分析 ===')
with MT5DataFetcher() as fetcher:
    mgr = MT5DataManager(fetcher)
    mgr.load()
    feat = mgr.feat_tensor
    syms = mgr.symbols

vm = StackVM()
factor = vm.execute(tokens, feat)
pos = torch.sign(torch.tanh(factor))
T = pos.shape[1]

total_trades = 0
for i, sym in enumerate(syms):
    p = pos[i]
    trades, runs, cur_len, cur_dir, prev = 0, [], 0, 0, 0
    for v in p.tolist():
        vi = int(v)
        if vi != 0:
            if vi == cur_dir:
                cur_len += 1
            else:
                if cur_len > 0: runs.append(cur_len)
                cur_dir, cur_len = vi, 1
                if prev != vi: trades += 1
        else:
            if cur_len > 0: runs.append(cur_len)
            cur_dir, cur_len = 0, 0
        prev = vi
    if cur_len > 0: runs.append(cur_len)
    avg_hold = sum(runs) / len(runs) if runs else 0
    total_trades += trades
    zero_pct = (p == 0).float().mean().item() * 100
    print(f'  {sym}: {trades}笔  每100bar={trades/T*100:.1f}笔  '
          f'均持仓={avg_hold:.1f}bar  空仓={zero_pct:.1f}%  '
          f'多/空={(p==1).sum().item()}/{(p==-1).sum().item()}')

avg100 = total_trades / (T * len(syms)) * 100
print(f'  三品种均: 每100bar={avg100:.1f}笔  每天~{avg100*24/100:.1f}笔')
