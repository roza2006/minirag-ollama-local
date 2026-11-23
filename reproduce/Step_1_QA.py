"""Step 1: answer the LiHua-World QA set with the index built by Step 0.

Usage (from the repo root):
    py -3.12 -u reproduce\\Step_1_QA.py                  # all 200 questions
    py -3.12 -u reproduce\\Step_1_QA.py --limit 3        # smoke test
    ... or double-click run_qa.bat

Writes one row per question to --outputpath, appending, so an interrupted
run can simply be restarted (already-answered questions are skipped).
"""

import argparse
import csv
import os
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

MODEL_ALIASES = {
    "QWEN3": "qwen3:8b",
    "PHI": "phi3.5",
    "GLM": "glm-edge",
    "MiniCPM": "minicpm3",
    "qwen": "qwen2.5:3b",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG QA (Ollama)")
    parser.add_argument("--model", type=str, default="QWEN3",
                        help="QWEN3 (默认，与索引一致) | PHI | GLM | MiniCPM | qwen")
    parser.add_argument("--llm", type=str, default=None)
    parser.add_argument("--mode", type=str, default="mini",
                        help="mini | light | naive")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument("--querypath", type=str,
                        default="./dataset/LiHua-World/qa/query_set.csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="only answer the first N questions (0 = all)")
    parser.add_argument("--topk", type=int, default=20,
                        help="检索返回条数。上游默认 60，上下文会涨到 6 万字符（约 1.5 万 token），"
                             "超过 Ollama 默认 4096 窗口被静默截断；20 约 4.8k token，能完整送进模型")
    parser.add_argument("--hybrid", type=int, default=0,
                        help="第二段检索：图路径之外再用原始问题做一次块向量检索，"
                             "补 N 条没进上下文的原文块。0=关闭（上游行为）")
    parser.add_argument("--response-type", type=str, default=None,
                        help="写进提示词的目标答案格式。上游默认 'Multiple Paragraphs'，"
                             "会让模型写大段散文；改成要求一两句话直接作答通常更准")
    parser.add_argument("--bm25", type=int, default=0,
                        help="BM25 稀疏补召回：用关键词精确匹配另取一批候选块，与图检索 RRF 融合，"
                             "截断回原条数（上下文体积不变）。0=关闭")
    parser.add_argument("--rerank", type=int, default=0,
                        help="块级重排：对全部候选块按与问题的相似度排序，只保留最相关的 K 条。"
                             "0=关闭（上游行为，候选全塞进上下文）")
    parser.add_argument("--max-async", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = get_args()

    llm_model = args.llm or MODEL_ALIASES.get(args.model)
    if not llm_model:
        log(f"Invalid model name: {args.model}")
        return 1

    workdir = os.path.abspath(args.workingdir)
    query_path = os.path.abspath(args.querypath)
    output_path = os.path.abspath(args.outputpath)

    if not os.path.isdir(workdir):
        log(f"!! index dir not found: {workdir}  (run Step 0 first)")
        return 1
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    log("=" * 70)
    log(f"LLM          : {llm_model}")
    log(f"endpoint     : {os.environ['OPENAI_API_BASE']}")
    log(f"working dir  : {workdir}")
    log(f"query set    : {query_path}")
    log(f"output       : {output_path}")
    log(f"query mode   : {args.mode}")
    log("=" * 70)

    log("[1/4] importing minirag / torch ...")
    from transformers import AutoModel, AutoTokenizer

    from minirag import MiniRAG, QueryParam
    from minirag.llm import gpt_4o_mini_complete, hf_embed
    from minirag.utils import EmbeddingFunc

    log("[2/4] loading embedding model from local cache ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

    rag = MiniRAG(
        working_dir=workdir,
        llm_model_func=gpt_4o_mini_complete,
        llm_model_max_token_size=32768,
        llm_model_max_async=args.max_async,
        llm_model_name=llm_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=512,
            func=lambda texts: hf_embed(
                texts, tokenizer=tokenizer, embed_model=embed_model
            ),
        ),
    )

    questions, gold_answers = [], []
    with open(query_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            questions.append(row["Question"])
            gold_answers.append(row["Gold Answer"])

    if args.limit:
        questions = questions[: args.limit]
        gold_answers = gold_answers[: args.limit]

    answered = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                answered.add(row["Question"])

    todo = [(q, a) for q, a in zip(questions, gold_answers) if q not in answered]
    log(f"[3/4] {len(questions)} questions, {len(answered)} already answered, "
        f"{len(todo)} to go")

    new_file = not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["Question", "Gold Answer", "minirag"])

        started = time.time()
        for i, (question, gold) in enumerate(todo, start=1):
            log("")
            log(f"--- Q{i}/{len(todo)}  {question}")
            log(f"    gold: {gold}")
            t0 = time.time()
            try:
                answer = rag.query(
                    question,
                    param=QueryParam(mode=args.mode, top_k=args.topk,
                                     hybrid_extra=args.hybrid,
                                     rerank_top=args.rerank,
                                     bm25_top=args.bm25,
                                     **({"response_type": args.response_type}
                                        if args.response_type else {})),
                ).replace("\n", " ").replace("\r", " ")
            except Exception as exc:
                log(f"    !! query failed: {type(exc).__name__}: {exc}")
                answer = "Error"
            log(f"    answer ({time.time() - t0:.1f}s): {answer[:300]}")
            writer.writerow([question, gold, answer])
            fh.flush()
            avg = (time.time() - started) / i
            log(f"    avg {avg:.1f}s/question, "
                f"ETA {(len(todo) - i) * avg / 60:.0f} min")

    log("")
    log(f"[4/4] answers written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
