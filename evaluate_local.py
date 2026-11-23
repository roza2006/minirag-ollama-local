"""用本地大模型给 QA 结果打分（替代 evaluation.ipynb 里的 GPT-4 裁判）。

    py -3.12 -u evaluate_local.py                        # 评 logs/Default_output.csv
    py -3.12 -u evaluate_local.py --input logs/qa_test5.csv --llm qwen3:8b

打分规则与仓库 evaluation.ipynb 一致：
    1  = 答对
    0  = 答非所问 / 没有回答
   -1  = 答错（与 gold 矛盾）
输出：逐题分数写回 csv，最后打印准确率（score==1 的比例）。
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("OLLAMA_NUM_CTX", "4096")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json_repair

from minirag.llm.openai import gpt_4o_mini_complete

JUDGE_PROMPT = """You are a strict grader. Compare the student's answer with the gold answer.

Question:
{question}

Gold Answer:
{gold}

Student Answer:
{answer}

Award the score by MEANING, not by wording:
- score  1  only if the student's answer states the same fact as the gold answer.
- score  0  if the answer is off-topic, evasive, or gives no usable answer.
- score -1  if the answer denies or contradicts the gold answer.

Judge this specific pair. Do not copy any example. Answer with JSON that has
two fields, "reason" (one short sentence) and "score" (an integer)."""


class FakeCache:
    global_config = {"llm_model_name": "qwen3:8b"}

    async def get_by_id(self, _k):
        return None

    async def upsert(self, _d):
        return None


def parse_score(raw: str) -> int:
    try:
        data = json_repair.loads(raw)
        if isinstance(data, dict):
            for key in ("score", "Score", "Score1"):
                if key in data:
                    return int(data[key])
    except Exception:
        pass
    low = (raw or "").lower()
    if '"score": 1' in low.replace(" ", "") or low.strip() == "1":
        return 1
    if "-1" in low:
        return -1
    if "0" in low:
        return 0
    return 0


async def judge(llm, question, gold, answer):
    cache = FakeCache()
    cache.global_config = {"llm_model_name": llm}
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=answer)
    out = await gpt_4o_mini_complete(prompt, hashing_kv=cache, max_tokens=200)
    return parse_score(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./logs/Default_output.csv")
    ap.add_argument("--output", default="./logs/Default_scores.csv")
    ap.add_argument("--llm", default="qwen3:8b")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.input, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"!! {args.input} 是空的，先跑 Step_1_QA 生成答案")
        return 1

    print(f"裁判模型 : {args.llm}")
    print(f"输入     : {args.input}（{len(rows)} 题）")
    print(f"输出     : {args.output}")
    print("-" * 60)

    scored = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        q = row["Question"]
        gold = row["Gold Answer"]
        ans = row.get("minirag") or row.get("answer") or ""
        s = await judge(args.llm, q, gold, ans)
        scored.append({**row, "score": s})
        right = sum(1 for r in scored if r["score"] == 1)
        print(f"{i:3d}. score={s:+d}  累计 {right}/{i} = {right / i * 100:5.1f}%   "
              f"{q[:60]}")

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    n = len(scored)
    right = sum(1 for r in scored if r["score"] == 1)
    wrong = sum(1 for r in scored if r["score"] == -1)
    zero = n - right - wrong
    print("-" * 60)
    print(f"题数     : {n}")
    print(f"答对 (1) : {right}  ({right / n * 100:.1f}%)")
    print(f"答非所问0: {zero}  ({zero / n * 100:.1f}%)")
    print(f"答错(-1) : {wrong}  ({wrong / n * 100:.1f}%)")
    print(f"耗时     : {(time.time() - t0) / 60:.1f} 分钟")
    print(f"明细     : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
