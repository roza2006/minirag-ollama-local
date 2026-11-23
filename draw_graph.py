"""把建好的知识图谱画成一张 PNG，方便直接看。"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# 中文字体，否则标题会变成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK = "LiHua-World"
OUT = os.path.join(WORK, "知识图谱.png")

G = nx.read_graphml(os.path.join(WORK, "graph_chunk_entity_relation.graphml"))
print(f"实体 {G.number_of_nodes()}，关系 {G.number_of_edges()}")

# 只画最大的连通分量，否则会有一堆孤立点挤成毛线球
if G.number_of_nodes():
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    biggest = G.subgraph(comps[0]).copy()
    print(f"最大连通分量: {biggest.number_of_nodes()} 实体 / "
          f"{biggest.number_of_edges()} 关系（覆盖 "
          f"{len(comps[0]) / G.number_of_nodes() * 100:.0f}%）")
else:
    biggest = G

degree = dict(biggest.degree())
top_nodes = sorted(degree, key=degree.get, reverse=True)[:22]

type_colors = {
    "PERSON": "#4C72B0", "ORGANIZATION": "#DD8452", "LOCATION": "#55A868",
    "EVENT": "#C44E52", "DATE": "#8172B3", "TIME": "#937860",
    "CONCEPT": "#DA8BC3", "OTHER": "#8C8C8C",
}
colors, sizes = [], []
for n, data in biggest.nodes(data=True):
    t = str(data.get("entity_type", "")).strip('"').upper()
    colors.append(type_colors.get(t, "#8C8C8C"))
    sizes.append(40 + degree[n] * 26)

plt.figure(figsize=(19, 13), facecolor="white")
pos = nx.spring_layout(biggest, k=0.42, iterations=90, seed=7)

nx.draw_networkx_edges(biggest, pos, alpha=0.16, width=0.7, edge_color="#666666")
nx.draw_networkx_nodes(biggest, pos, node_color=colors, node_size=sizes,
                       linewidths=0.5, edgecolors="white", alpha=0.92)
nx.draw_networkx_labels(biggest, pos, labels={n: n.strip('"') for n in top_nodes},
                        font_size=9, font_weight="bold")

handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=k)
           for k, c in type_colors.items()]
plt.legend(handles=handles, loc="upper right", fontsize=11, title="实体类型",
           frameon=True, title_fontsize=12)
plt.title(f"LiHua-World 知识图谱（{biggest.number_of_nodes()} 实体 / "
          f"{biggest.number_of_edges()} 关系，只画最大连通分量）", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("已保存:", OUT, f"{os.path.getsize(OUT)/1024:.0f} KB")
