"""把 MiniRAG 的 graphml 导出成 yED 能直接打开、且有正常布局的版本。

修过两个问题：
  1. 原文件的 <node id> 带引号（"ALICE" 形式），不是合法的 NMTOKEN，
     yED 做 schema 校验时直接拒绝打开；networkx 宽容所以 Python 侧没暴露。
  2. 第一版导出只写了 y:Geometry 的宽高、没写 x/y，结果所有节点堆在原点。
     现在用 networkx 的力导向布局预先把坐标算好写进去。

用法：
    py -3.12 -u export_for_yed.py                  # 全部实体
    py -3.12 -u export_for_yed.py --top 200        # 只导连接最多的 200 个
    py -3.12 -u export_for_yed.py --drop-event     # 去掉 EVENT 类节点（占 47%）
"""

import argparse
import math
import re
import xml.sax.saxutils as su

import networkx as nx

SRC = "LiHua-World/graph_chunk_entity_relation.graphml"
OUT = "LiHua-World/知识图谱_yED.graphml"
YNS = "{http://www.yworks.com/xml/graphml}"

TYPE_COLORS = {
    "PERSON": "#4C72B0",
    "ORGANIZATION": "#DD8452",
    "LOCATION": "#55A868",
    "EVENT": "#C44E52",
    "DATE": "#8172B3",
    "TIME": "#937860",
    "CONCEPT": "#DA8BC3",
    "BAND NAME": "#E17C05",
    "TECHNOLOGY": "#2CA02C",
}
DEFAULT_COLOR = "#8C8C8C"
INVALID = re.compile(r"[^A-Za-z0-9_.:\-]")
LABEL_MAX = 26


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


def label_of(name):
    text = str(name).strip().strip('"')
    return text if len(text) <= LABEL_MAX else text[: LABEL_MAX - 1] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="只导出度最高的 N 个实体")
    ap.add_argument("--drop-event", action="store_true", help="去掉 EVENT 类节点")
    ap.add_argument("--no-layout", action="store_true", help="不预计算坐标")
    args = ap.parse_args()

    G = nx.read_graphml(SRC)

    if args.drop_event:
        drop = [n for n, d in G.nodes(data=True)
                if str(d.get("entity_type", "")).strip('"').upper() == "EVENT"]
        G.remove_nodes_from(drop)
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
        print(f"去掉 {len(drop)} 个 EVENT 节点后剩 {G.number_of_nodes()} 个")

    if args.top and G.number_of_nodes() > args.top:
        keep = sorted(G.degree, key=lambda kv: kv[1], reverse=True)[: args.top]
        G = G.subgraph([n for n, _ in keep]).copy()
        print(f"只保留度最高的 {args.top} 个实体")

    if args.no_layout:
        pos = {n: (0.0, 0.0) for n in G.nodes}
    else:
        n = max(G.number_of_nodes(), 2)
        pos = nx.spring_layout(G, k=6.0 / math.sqrt(n), iterations=120, seed=20261123)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        span_x = (max(xs) - min(xs)) or 1.0
        span_y = (max(ys) - min(ys)) or 1.0
        pos = {k: ((v[0] - min(xs)) / span_x * 6000.0,
                   (v[1] - min(ys)) / span_y * 4000.0) for k, v in pos.items()}

    used, id_of = set(), {}
    for node in G.nodes:
        id_of[node] = clean_id(node, used)

    deg = dict(G.degree)
    max_deg = max(deg.values()) if deg else 1

    head = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:y="http://www.yworks.com/xml/graphml"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">',
        '  <key id="d_type" for="node" attr.name="entity_type" attr.type="string"/>',
        '  <key id="d_desc" for="node" attr.name="description" attr.type="string"/>',
        '  <key id="d_deg"  for="node" attr.name="degree" attr.type="int"/>',
        '  <key id="d_ngfx" for="node" yfiles.type="nodegraphics"/>',
        '  <key id="d_edesc" for="edge" attr.name="description" attr.type="string"/>',
        '  <key id="d_egfx" for="edge" yfiles.type="edgegraphics"/>',
        '  <graph id="minirag" edgedefault="directed">',
    ]

    body = []
    for node, data in G.nodes(data=True):
        etype = str(data.get("entity_type", "")).strip('"').upper()
        color = TYPE_COLORS.get(etype, DEFAULT_COLOR)
        size = 30 + 46 * deg[node] / max_deg if max_deg else 30
        x = pos[node][0] - size / 2
        y = pos[node][1] - size / 2
        desc = str(data.get("description", ""))[:3000]

        body += [
            f'    <node id="{id_of[node]}">',
            f'      <data key="d_type">{su.escape(etype)}</data>',
            f'      <data key="d_desc">{su.escape(desc)}</data>',
            f'      <data key="d_deg">{deg[node]}</data>',
            '      <data key="d_ngfx">',
            '        <y:ShapeNode>',
            f'          <y:Geometry x="{x:.1f}" y="{y:.1f}"'
            f' height="{size:.1f}" width="{size:.1f}"/>',
            f'          <y:Fill color="{color}" transparent="false"/>',
            '          <y:BorderStyle color="#FFFFFF" type="line" width="1.0"/>',
            '          <y:NodeLabel alignment="center" autoSizePolicy="content"'
            ' fontSize="11" fontName="Dialog" fontStyle="bold"'
            ' textColor="#1A1A1A" visible="true" hasBackground="false">',
            f'            {su.escape(label_of(node))}',
            '          </y:NodeLabel>',
            '          <y:Shape type="ellipse"/>',
            '        </y:ShapeNode>',
            '      </data>',
            '    </node>',
        ]

    for i, (u, v, data) in enumerate(G.edges(data=True), start=1):
        desc = str(data.get("description", ""))[:1500]
        body += [
            f'    <edge id="e{i}" source="{id_of[u]}" target="{id_of[v]}">',
            f'      <data key="d_edesc">{su.escape(desc)}</data>',
            '      <data key="d_egfx">',
            '        <y:PolyLineEdge>',
            '          <y:LineStyle color="#BDBDBD" type="line" width="1.0"/>',
            '          <y:Arrows source="none" target="standard"/>',
            '          <y:BendStyle smoothed="false"/>',
            '        </y:PolyLineEdge>',
            '      </data>',
            '    </edge>',
        ]

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + body + ["  </graph>", "</graphml>"]))

    import xml.etree.ElementTree as ET

    NS = "{http://graphml.graphdrawing.org/xmlns}"
    root = ET.parse(OUT).getroot()
    ids = [e.get("id") for e in root.iter(f"{NS}node")]
    bad = [x for x in ids if not re.match(r"^[A-Za-z_][A-Za-z0-9_.:\-]*$", x)]
    coords = [(d.get("x"), d.get("y")) for d in root.iter(YNS + "Geometry")]
    print(f"已写出 {OUT}")
    print(f"  节点 {len(ids)}  边 {sum(1 for _ in root.iter(f'{NS}edge'))}")
    print(f"  非法 id {len(bad)}   重复坐标 {len(coords) - len(set(coords))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
