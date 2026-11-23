"""在同一批题上对比多个配置的答对率。

用法：
    py -3.12 -u compare_ab.py 基线=logs/Default_scores.csv bm25=logs/ab_bm25_scores.csv
    py -3.12 -u compare_ab.py --subset dataset/LiHua-World/qa/subset_bm25.csv 基线=... ...

默认用 subset60；--subset 可以换成别的题目集合，保证比的是同一批题。
"""
import csv
import os
import sys

SUBSET = "./dataset/LiHua-World/qa/subset60.csv"

args = sys.argv[1:]
if "--subset" in args:
    i = args.index("--subset")
    SUBSET = args[i + 1]
    args = args[:i] + args[i + 2:]

keys = []
with open(SUBSET, encoding="utf-8-sig") as fh:
    for row in csv.DictReader(fh):
        keys.append(row["Question"])
key_set = set(keys)

results = {}
for spec in args:
    if "=" not in spec:
        continue
    name, path = spec.split("=", 1)
    if not os.path.exists(path):
        print(f"!! 跳过 {name}: {path} 不存在")
        continue
    scores = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["Question"] in key_set:
                scores[row["Question"]] = int(row["score"])
    n = len(scores)
    if n == 0:
        print(f"!! 跳过 {name}: 没匹配到子集里的题")
        continue
    right = sum(1 for v in scores.values() if v == 1)
    zero = sum(1 for v in scores.values() if v == 0)
    wrong = sum(1 for v in scores.values() if v == -1)
    results[name] = (n, right, zero, wrong)

print(f"{'配置':<14}{'题数':>6}{'答对':>8}{'答对率':>9}{'答非所问':>10}{'答错':>8}")
print("-" * 58)
for name, (n, right, zero, wrong) in results.items():
    print(f"{name:<14}{n:>6}{right:>8}{right / n * 100:>8.1f}%{zero:>10}{wrong:>8}")
