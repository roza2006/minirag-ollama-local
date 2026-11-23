"""Step 0: build the MiniRAG index for LiHua-World with a local Ollama model.

Usage (from the repo root):
    py -3.12 -u reproduce\\Step_0_index.py                 # index everything
    py -3.12 -u reproduce\\Step_0_index.py --limit 5       # smoke test
    ... or just double-click run_index.bat

Everything is local: LLM = Ollama phi3.5 through the OpenAI-compatible
/v1 endpoint, embeddings = all-MiniLM-L6-v2 (384-d) from the local
HuggingFace cache. No OpenAI account, no network needed.
"""

import argparse
import logging
import os
import sys
import time

# --------------------------------------------------------------------------
# 0. Console + environment, before any heavy import
# --------------------------------------------------------------------------
# A GBK console raises UnicodeEncodeError on emoji / on log lines with
# non-GBK characters, and that used to kill the run with no useful message.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("LLM_BASE_URL", os.environ["OPENAI_API_BASE"])
# The MiniLM weights are already in the local HF cache; never try to phone home.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

MODEL_ALIASES = {
    "PHI": "phi3.5",
    "GLM": "glm-edge",
    "MiniCPM": "minicpm3",
    "qwen": "qwen2.5:3b",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG index builder (Ollama)")
    parser.add_argument("--model", type=str, default="PHI",
                        help="PHI | GLM | MiniCPM | qwen, or any Ollama tag via --llm")
    parser.add_argument("--llm", type=str, default=None,
                        help="explicit Ollama model tag, e.g. phi3.5 / qwen3:8b")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument("--querypath", type=str,
                        default="./dataset/LiHua-World/qa/query_set.csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="only index the first N documents (0 = all)")
    parser.add_argument("--batch", type=int, default=8,
                        help="documents per insert() call (default 8)")
    parser.add_argument("--max-async", type=int, default=4,
                        help="concurrent LLM calls (default 4, keeps a single "
                             "Ollama instance from thrashing)")
    parser.add_argument("--gleaning", type=int, default=1,
                        help="extra extraction passes per chunk (1 = upstream "
                             "default; 0 halves the LLM calls at ~90 min -> ~45 min "
                             "for the full set, with slightly lower recall)")
    return parser.parse_args()


def main() -> int:
    args = get_args()

    if args.llm:
        llm_model = args.llm
    elif args.model in MODEL_ALIASES:
        llm_model = MODEL_ALIASES[args.model]
    else:
        log(f"Invalid model name: {args.model}")
        return 1

    workdir = os.path.abspath(args.workingdir)
    data_path = os.path.abspath(args.datapath)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    log("=" * 70)
    log(f"LLM            : {llm_model}")
    log(f"LLM endpoint   : {os.environ['OPENAI_API_BASE']}")
    log(f"embedding      : {EMBEDDING_MODEL} ({EMBEDDING_DIM}-d, offline)")
    log(f"working dir    : {workdir}")
    log(f"data dir       : {data_path}")
    log("=" * 70)

    # Cheap preflight: is Ollama actually serving that model?
    import urllib.request
    import json as _json
    try:
        with urllib.request.urlopen(
            os.environ["OPENAI_API_BASE"].rstrip("/") + "/models", timeout=5
        ) as r:
            served = [m["id"] for m in _json.load(r)["data"]]
        log(f"Ollama serves  : {', '.join(served) or '(none)'}")
        if llm_model not in served and f"{llm_model}:latest" not in served:
            log(f"!! '{llm_model}' is not in `ollama list` -> run: ollama pull {llm_model}")
            return 1
    except Exception as exc:
        log(f"!! cannot reach Ollama at {os.environ['OPENAI_API_BASE']}: {exc}")
        log("   start it with:  ollama serve")
        return 1

    # ---------------------------------------------------------------- imports
    log("[1/4] importing minirag / torch ... (10-20s the first time)")
    t0 = time.time()
    from transformers import AutoModel, AutoTokenizer

    from minirag import MiniRAG
    from minirag.llm import gpt_4o_mini_complete, hf_embed
    from minirag.utils import EmbeddingFunc

    log(f"      imports done in {time.time() - t0:.1f}s")

    # One INFO line per HTTP request drowns out the real progress.
    for _noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "nano-vectordb",
                   "urllib3", "transformers"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    log("[2/4] loading embedding model from local cache ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    log(f"      loaded in {time.time() - t0:.1f}s")

    rag = MiniRAG(
        working_dir=workdir,
        llm_model_func=gpt_4o_mini_complete,
        llm_model_max_token_size=32768,
        llm_model_max_async=args.max_async,
        entity_extract_max_gleaning=args.gleaning,
        llm_model_name=llm_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=512,
            func=lambda texts: hf_embed(
                texts, tokenizer=tokenizer, embed_model=embed_model
            ),
        ),
    )
    log("      MiniRAG initialised")

    # ------------------------------------------------------------- documents
    files = []
    for root, _dirs, names in os.walk(data_path):
        for name in sorted(names):
            if name.endswith(".txt"):
                files.append(os.path.join(root, name))
    files.sort()

    if args.limit:
        files = files[: args.limit]

    log(f"[3/4] found {len(files)} txt documents, batching by {args.batch}")
    if not files:
        log(f"!! no .txt under {data_path}")
        return 1

    batch_size = max(1, args.batch)
    started = time.time()
    done = 0
    for start in range(0, len(files), batch_size):
        batch = files[start: start + batch_size]
        texts = []
        for path in batch:
            try:
                with open(path, encoding="utf-8") as fh:
                    texts.append(fh.read())
            except UnicodeDecodeError:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    texts.append(fh.read())

        done += len(batch)
        elapsed = time.time() - started
        speed = done / elapsed if elapsed else 0
        eta = (len(files) - done) / speed if speed else 0
        log("")
        log(f"--- batch {start // batch_size + 1}/"
            f"{(len(files) + batch_size - 1) // batch_size} "
            f"| {done}/{len(files)} docs "
            f"| {elapsed / 60:.1f} min elapsed "
            f"| {speed * 60:.1f} docs/min "
            f"| ETA {eta / 60:.0f} min")
        log(f"    {os.path.basename(batch[0])} ... {os.path.basename(batch[-1])}")

        t_batch = time.time()
        rag.insert(texts)
        log(f"    batch indexed in {time.time() - t_batch:.1f}s")

    log("")
    log(f"[4/4] done: {done} documents in {(time.time() - started) / 60:.1f} min")
    log(f"      index files are in {workdir}")
    return 0


if __name__ == "__main__":
    logging.getLogger("minirag").setLevel(logging.INFO)
    sys.exit(main())
