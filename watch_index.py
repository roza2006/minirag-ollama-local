"""盯着 MiniRAG 索引，一跑完就弹窗 + 响铃，不用一直盯着看。

    watch_index.bat            正常用，后台每 60 秒查一次
    watch_index.bat --once     只查一次，打印结论（不弹窗）
    watch_index.bat --test     测试弹窗能不能正常出现（3 秒后自动关）

判定逻辑：
  * logs\\qwen_out.log 里出现 "[4/4] done:"          -> 正常完成
  * python 进程没了、也不见了完成标志                -> 中途断了
  * 超过 6 小时还没结束                              -> 放弃并提示
"""
import json
import os
import re
import subprocess
import sys
import threading
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
LOG_WATCH = os.path.join(ROOT, "logs", "watch_index.log")

POLL_SECONDS = 60
MAX_HOURS = 6


def log(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {msg}"
    print(line, flush=True)
    try:
        with open(LOG_WATCH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def indexer_running():
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return False
    return "Step_0_index" in out


def finished_line():
    try:
        with open(LOG_OUT, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "[4/4] done:" in line:
                    return line.strip()
    except Exception:
        return None
    return None


def stats():
    def load(name):
        try:
            with open(os.path.join(WORK, name), encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    chunks = load("kv_store_text_chunks.json")
    ex = load("kv_store_extracted_chunks.json")
    done = sum(1 for v in ex.values() if isinstance(v, dict) and v.get("done"))
    nodes = edges = "?"
    try:
        import networkx as nx
        g = nx.read_graphml(os.path.join(WORK, "graph_chunk_entity_relation.graphml"))
        nodes, edges = g.number_of_nodes(), g.number_of_edges()
    except Exception:
        pass
    pct = (done * 100 // len(chunks)) if chunks else 0
    return (f"文档 442 篇\n"
            f"分片 {done} / {len(chunks)} （{pct}%）\n"
            f"知识图谱 {nodes} 个实体 / {edges} 条关系")


def notify(title, body, seconds=None):
    """弹一个置顶窗口并响铃；seconds 给定时自动关闭（用于自测）。"""
    try:
        import winsound
        for freq in (880, 1320, 1760):
            winsound.Beep(freq, 180)
    except Exception:
        pass

    try:
        import ctypes
        MB_TOPMOST = 0x00040000
        MB_SETFOREGROUND = 0x00010000
        MB_ICONINFORMATION = 0x00000040

        if seconds:
            def closer():
                time.sleep(seconds)
                hwnd = ctypes.windll.user32.FindWindowW(None, title)
                if hwnd:
                    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            threading.Thread(target=closer, daemon=True).start()

        ctypes.windll.user32.MessageBoxW(
            None, body, title, MB_TOPMOST | MB_SETFOREGROUND | MB_ICONINFORMATION
        )
    except Exception as exc:
        log(f"弹窗失败 {exc}")


def main():
    once = "--once" in sys.argv
    test = "--test" in sys.argv

    if test:
        print("3 秒后自动关闭的测试弹窗……")
        notify("MiniRAG 通知测试", "看到这个窗口说明弹窗正常。\n3 秒后自动关闭。", seconds=3)
        print("测试结束。")
        return 0

    if once:
        fin = finished_line()
        run = indexer_running()
        print(f"索引进程: {'在跑' if run else '没在跑'}")
        print(f"完成标志: {fin or '没有'}")
        print(stats())
        return 0 if fin else 1

    log("开始盯索引（每 60 秒查一次）")
    started = time.time()

    while True:
        fin = finished_line()
        if fin:
            log(f"完成：{fin}")
            notify("MiniRAG 索引跑完了", f"{fin}\n\n{stats()}\n\n下一步：run_qa.bat --limit 5")
            return 0

        if not indexer_running():
            # 进程可能是刚被重启（改代码后重跑），先给 2 分钟宽限再报警
            log("进程暂时不见了，等 120 秒确认是不是在重启")
            time.sleep(120)
            if finished_line():
                continue
            if indexer_running():
                log("重新出现了，是重启，继续盯")
                continue
            log("确认停止且没有完成标志 -> 中途断了")
            notify("MiniRAG 索引中断了",
                   "进程没了，但没看到 [4/4] done。\n\n"
                   f"{stats()}\n\n下一步：重跑 run_index.bat --llm qwen3:8b --gleaning 0\n"
                   "（已抽到的不会重做）")
            return 1

        if time.time() - started > MAX_HOURS * 3600:
            log(f"超过 {MAX_HOURS} 小时仍未结束，放弃监视")
            notify("MiniRAG 监视超时",
                   f"已经盯了 {MAX_HOURS} 小时还没跑完。\n\n{stats()}")
            return 2

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
