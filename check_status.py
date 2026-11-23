"""一句话看索引跑到哪了 / 跑完没有。双击 check_status.bat 或直接运行本文件。"""
import json
import os
import re
import subprocess
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "LiHua-World")
LOG_OUT = os.path.join(ROOT, "logs", "qwen_out.log")
LOG_ERR = os.path.join(ROOT, "logs", "qwen_err.log")
TICK = re.compile("[\u2800-\u28ff]")


def load(name, default=None):
    try:
        with open(os.path.join(WORK, name), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def running_python():
    """Return the PIDs of the python processes that belong to this project."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get",
             "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        if "Step_0_index" in line or "Step_1_QA" in line:
            m = re.search(r",(\d+)\s*$", line.strip())
            if m:
                pids.append(m.group(1))
    return pids


def last_line(path):
    """Last meaningful line of a log whose progress ticker uses bare \\r."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 6000))
            text = fh.read().decode("utf-8", "replace")
    except Exception:
        return "", ""
    parts = [p.strip() for p in re.split(r"[\r\n]+", text) if p.strip()]
    last = ""
    for part in reversed(parts):
        part = TICK.sub("", part).strip()
        if part and not part.startswith(("Generating embeddings", "Loading weights")):
            last = part
            break
    return last, text


def ticker_count(text):
    found = re.findall(r"Processed (\d+) chunks", text or "")
    return int(found[-1]) if found else 0


def count_warnings(path, needle):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if needle in line)
    except Exception:
        return 0


def main():
    docs = load("kv_store_doc_status.json", {}) or {}
    chunks = load("kv_store_text_chunks.json", {}) or {}
    ex = load("kv_store_extracted_chunks.json", {}) or {}
    done = sum(1 for k, v in ex.items()
               if k in chunks and isinstance(v, dict) and v.get("done"))

    nodes = edges = None
    try:
        import networkx as nx
        g = nx.read_graphml(os.path.join(WORK, "graph_chunk_entity_relation.graphml"))
        nodes, edges = g.number_of_nodes(), g.number_of_edges()
    except Exception:
        pass

    pids = running_python()
    finished = None
    try:
        with open(LOG_OUT, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "[4/4] done:" in line:
                    finished = line.strip()
    except Exception:
        pass

    print()
    print("=" * 64)
    print("  MiniRAG 索引状态")
    print("=" * 64)
    print(f"  进程        : {'在跑 PID ' + ','.join(pids) if pids else '没有在跑'}")
    print(f"  文档        : {len(docs)} 篇已处理")
    print(f"  分片        : {len(chunks)} 个 chunk")
    pct = (done * 100 // len(chunks)) if chunks else 0
    bar_len = 30
    filled = int(bar_len * pct / 100)
    print(f"  已抽到实体  : {done} / {len(chunks)}  "
          f"[{'#' * filled}{'.' * (bar_len - filled)}] {pct}%")
    if nodes is not None:
        print(f"  知识图谱    : {nodes} 个实体, {edges} 条关系")

    tail, tail_text = last_line(LOG_OUT)
    if tail:
        print(f"  最新输出    : {tail[:90]}")
    progress_now = ticker_count(tail_text)
    if progress_now and not finished:
        print(f"  本轮已抽    : {progress_now} 个 chunk"
              f"（每 64 个才落盘一次，所以上面的百分比是滞后的）")

    gave_up = count_warnings(LOG_ERR, "gave up on a repeating")
    if gave_up:
        print(f"  模型复读放弃: {gave_up} 次（phi3.5 的老问题，qwen3 基本不犯）")

    print("-" * 64)
    if finished:
        print(f"  判定        : 跑完了 -> {finished}")
        print("  下一步      : run_qa.bat --limit 5   看 5 题答得像不像，再 run_qa.bat 全量")
    elif pids:
        if chunks:
            left = len(chunks) - done
            print(f"  判定        : 还在跑，还剩约 {left} 个 chunk")
            print("  下一步      : 等。每 64 个 chunk 存一次档，中断了重跑同一命令会接着做")
    else:
        print("  判定        : 进程没了、也没看到完成标志 -> 中途断了")
        print("  下一步      : 重跑 run_index.bat --llm qwen3:8b --gleaning 0")
    print()


if __name__ == "__main__":
    main()
    if sys.stdin and sys.stdin.isatty():
        try:
            input("按回车关闭...")
        except (EOFError, KeyboardInterrupt):
            pass
