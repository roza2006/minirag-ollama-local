"""生成固定评测子集（分层抽样，可复现），供不同配置对比用。"""
import csv
import random

SRC = "dataset/LiHua-World/qa/query_set.csv"
OUT = "dataset/LiHua-World/qa/subset120.csv"
PER_TYPE = {"Single": 60, "Multi": 30, "Null": 30}

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
buckets = {}
for r in rows:
    buckets.setdefault(r["Type"], []).append(r)

random.seed(20261122)
picked = []
for t, n in PER_TYPE.items():
    bucket = buckets.get(t, [])
    picked.extend(random.sample(bucket, min(n, len(bucket))))

with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["Question", "Gold Answer", "Evidence", "Type"])
    w.writeheader()
    w.writerows(picked)

print(f"已写出 {OUT}，{len(picked)} 题")
for t in PER_TYPE:
    print(f"  {t}: {sum(1 for r in picked if r['Type'] == t)}")
