"""
train_next_groups.py — 训练剩余组（跳过已完成的 forex）

用法：
    python train_next_groups.py --offline

依次训练 metals_comm、index 组，使用 D:\K线数据 全量历史。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from main import train_group

# 跳过已完成的 forex 组
GROUPS_TO_TRAIN = ["metals_comm", "index"]


def main():
    offline = "--offline" in sys.argv
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  训练剩余组: {GROUPS_TO_TRAIN}")
    print(f"  跳过: forex（已完成，Best=0.485）")
    print(f"  offline={offline}  BARS_COUNT={Config.BARS_COUNT}")
    print(f"{'='*60}")

    results = {}
    with MT5DataFetcher(offline=offline) as fetcher:
        for gname in GROUPS_TO_TRAIN:
            gsyms = Config.SYMBOL_GROUPS.get(gname, [])
            if not gsyms:
                print(f"  [跳过] 组 {gname} 不存在")
                continue

            print(f"\n>>> 开始训练 [{gname}] 组: {gsyms}")
            eng = train_group(fetcher, gname, gsyms, offline)
            if eng is not None:
                results[gname] = {
                    "score": eng.best_score,
                    "formula": eng.best_formula,
                    "readable": eng._decode_formula(eng.best_formula),
                }
                print(f"<<< [{gname}] 完成: score={eng.best_score:.4f}")
                print(f"    {results[gname]['readable']}")
            else:
                print(f"<<< [{gname}] 失败")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  全部训练完成  耗时 {elapsed/3600:.2f}h")
    print(f"{'='*60}")
    for gname, r in results.items():
        print(f"  [{gname:12s}]  score={r['score']:.4f}")
        print(f"    {r['readable']}")


if __name__ == "__main__":
    main()
