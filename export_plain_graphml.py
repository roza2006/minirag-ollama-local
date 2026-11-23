"""最朴素的 GraphML 导出：只有 id + 属性，不带任何 yED 专用样式。

如果 知识图谱_yED.graphml 在你的 yED 版本里还是打不开，用这个。
它只依赖 GraphML 核心规范，Gephi / Cytoscape / yED / NetworkX 都能读。

用法：
    py -3.12 -u export_plain_graphml.py [--top N]
"""

import argparse
import re
import xml.sax.saxutils as su

import networkx as nx

SRC = "LiHua-World/graph_chunk_entity_relation.graphml"
OUT = "LiHua-World/知识图谱_plain.graphml"
INVALID = re.compile(r"[^A-Za-z0-9_.:\-]")


def clean_id(name, used):
    base = INVALID.sub("_", str(name).strip().strip('"').strip("'")).strip("_")
    if not base:
        base = "node"
    if not base[0].isalpha() and base[0] != "_":
        base = "N" + base
    base = base[:60]
    cand, i = base, 2
    while cand in used:
        cand = f"{base}_{i}"
        i += 1
    used.add(cand)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    G = nx.read_graphml(SRC)
    if args.top and G.number_of_nodes() > args.top:
        keep = sorted(G.degree, key=lambda kv: kv[1], reverse=True)[: args.top]
        G = G.subgraph([n for n, _ in keep]).copy()

    used, id_of = set(), {}
    for n in G.nodes:
        id_of[n] = clean_id(n, used)

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="k_name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="k_type" for="node" attr.name="entity_type" attr.type="string"/>',
        '  <key id="k_desc" for="node" attr.name="description" attr.type="string"/>',
        '  <key id="k_deg"  for="node" attr.name="degree" attr.type="int"/>',
        '  <key id="k_edesc" for="edge" attr.name="description" attr.type="string"/>',
        '  <graph id="minirag" edgedefault="directed">',
    ]

    deg = dict(G.degree)
    for n, data in G.nodes(data=True):
        out.append(f'    <node id="{id_of[n]}">')
        out.append(f'      <data key="k_name">{su.escape(str(n).strip(chr(34)))}</data>')
        out.append(f'      <data key="k_type">{su.escape(str(data.get("entity_type", "")).strip(chr(34)))}</data>')
        out.append(f'      <data key="k_desc">{su.escape(str(data.get("description", ""))[:3000])}</data>')
        out.append(f'      <data key="k_deg">{deg[n]}</data>')
        out.append('    </node>')

    for i, (u, v, data) in enumerate(G.edges(data=True), start=1):
        out.append(f'    <edge id="e{i}" source="{id_of[u]}" target="{id_of[v]}">')
        out.append(f'      <data key="k_edesc">{su.escape(str(data.get("description", ""))[:1500])}</data>')
        out.append('    </edge>')

    out += ['  </graph>', '</graphml>']

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))

    back = nx.read_graphml(OUT)
    print(f"已写出 {OUT}")
    print(f"  节点 {back.number_of_nodes()}  边 {back.number_of_edges()}")


if __name__ == "__main__":
    main()
