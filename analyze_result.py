"""按题型拆分评分结果（配合 evaluate_local.py 生成的结果 csv 使用）。

用法：py -3.12 -u analyze_result.py [--scores logs/Default_scores.csv]
"""
import argparse
import collections
import csv

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="./logs/Default_scores.csv")
ap.add_argument("--queries", default="./dataset/LiHua-World/qa/query_set.csv")
args = ap.parse_args()

qtype = {}
with open(args.queries, encoding="utf-8-sig") as fh:
    for row in csv.DictReader(fh):
        qtype[row["Question"]] = row["Type"]

by_type = collections.defaultdict(lambda: {"n": 0, "1": 0, "0": 0, "-1": 0})
total = {"n": 0, "1": 0, "0": 0, "-1": 0}
evade = 0

with open(args.scores, encoding="utf-8-sig") as fh:
    for row in csv.DictReader(fh):
        t = qtype.get(row["Question"], "?")
        s = str(int(row["score"]))
        by_type[t]["n"] += 1
        by_type[t][s] += 1
        total["n"] += 1
        total[s] += 1
        a = (row.get("minirag") or "").lower()
        if any(k in a for k in ("no mention", "no information", "insufficient",
                                "no evidence", "not able to provide", "not provided")):
            evade += 1

print(f"{'题型':<8}{'题数':>6}{'答对':>8}{'答对率':>9}{'答非所问':>10}{'答错':>8}")
print("-" * 52)
for t in sorted(by_type, key=lambda x: -by_type[x]["n"]):
    d = by_type[t]
    print(f"{t:<8}{d['n']:>6}{d['1']:>8}{d['1'] / d['n'] * 100:>8.1f}%"
          f"{d['0']:>10}{d['-1']:>8}")
print("-" * 52)
print(f"{'合计':<8}{total['n']:>6}{total['1']:>8}{total['1'] / total['n'] * 100:>8.1f}%"
      f"{total['0']:>10}{total['-1']:>8}")
print(f"\n回答里明确说\"没有相关信息\"的: {evade} 题 "
      f"({evade / max(total['n'], 1) * 100:.1f}%)")
