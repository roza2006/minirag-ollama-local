"""为 BM25 实验挑子集：答案是需要"字面精确匹配"的题（日期 / 短事实），
并把基线在全量 637 题上的结果切出来，避免重跑。

输出：
    dataset/LiHua-World/qa/subset_bm25.csv   子集题目
    logs/subset_bm25_baseline.csv            基线在这些题上的打分
"""

import csv
import re

SRC = "dataset/LiHua-World/qa/query_set.csv"
SCORES = "logs/Default_scores.csv"
OUT_SUB = "dataset/LiHua-World/qa/subset_bm25.csv"
OUT_BASE = "logs/subset_bm25_baseline.csv"

DATE = re.compile(r"20\d{6}")
SKIP = {"yes", "no", "insufficient information"}

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
picked = []
for r in rows:
    gold = (r["Gold Answer"] or "").strip()
    if gold.lower() in SKIP or not gold:
        continue
    if DATE.search(gold) or len(gold) <= 40:
        picked.append(r)

picked = picked[:90]
with open(OUT_SUB, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["Question", "Gold Answer", "Evidence", "Type"])
    w.writeheader()
    w.writerows(picked)

questions = {r["Question"] for r in picked}
scored = [r for r in csv.DictReader(open(SCORES, encoding="utf-8-sig"))
          if r["Question"] in questions]
with open(OUT_BASE, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(scored[0].keys()))
    w.writeheader()
    w.writerows(scored)

n = len(scored)
right = sum(1 for r in scored if int(r["score"]) == 1)
print(f"子集 {len(picked)} 题，其中已有基线打分的 {n} 题")
print(f"基线在这些题上的准确率：{right}/{n} = {right / n * 100:.1f}%")
print("（对比：全量 637 题 45.8%，60 题抽样 51.7%）")
