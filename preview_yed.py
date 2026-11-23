"""把导出的 yED graphml 实际渲染一遍，确认坐标真的散开了（不是堆在原点）。"""
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SRC = sys.argv[1] if len(sys.argv) > 1 else "LiHua-World/知识图谱_yED.graphml"
OUT = sys.argv[2] if len(sys.argv) > 2 else "LiHua-World/_yED预览.png"

GNS = "{http://graphml.graphdrawing.org/xmlns}"
YNS = "{http://www.yworks.com/xml/graphml}"

root = ET.parse(SRC).getroot()
pos, color, label, size = {}, {}, {}, {}
for node in root.iter(GNS + "node"):
    nid = node.get("id")
    for d in node.findall(GNS + "data"):
        if d.get("key") == "d_ngfx":
            geo = d.find(f"{YNS}ShapeNode/{YNS}Geometry")
            fill = d.find(f"{YNS}ShapeNode/{YNS}Fill")
            lbl = d.find(f"{YNS}ShapeNode/{YNS}NodeLabel")
            if geo is not None:
                w = float(geo.get("width", 30))
                x = float(geo.get("x", 0)) + w / 2
                y = float(geo.get("y", 0)) + w / 2
                pos[nid] = (x, -y)          # y 轴翻转，屏幕坐标
                size[nid] = w
            if fill is not None:
                color[nid] = fill.get("color", "#8C8C8C")
            if lbl is not None:
                label[nid] = (lbl.text or "").strip()

edges = [(e.get("source"), e.get("target")) for e in root.iter(GNS + "edge")]
print(f"节点 {len(pos)}  边 {len(edges)}")

xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]
print(f"坐标范围 x[{min(xs):.0f},{max(xs):.0f}]  y[{min(ys):.0f},{max(ys):.0f}]")
print(f"不同坐标点: {len(set(pos.values()))}  （= 节点数说明没有堆叠）")

fig, ax = plt.subplots(figsize=(20, 13), facecolor="white")
for u, v in edges:
    if u in pos and v in pos:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#CCCCCC", lw=0.5, alpha=0.7, zorder=1)
for nid, (x, y) in pos.items():
    ax.scatter(x, y, s=size.get(nid, 30) ** 1.6 / 4,
               c=color.get(nid, "#8C8C8C"), edgecolors="white",
               linewidths=0.6, zorder=2)

# 只给度数高的节点标名字，否则重叠不可读
deg = {}
for u, v in edges:
    deg[u] = deg.get(u, 0) + 1
    deg[v] = deg.get(v, 0) + 1
for nid, _ in sorted(deg.items(), key=lambda kv: -kv[1])[:28]:
    if nid in pos:
        ax.text(pos[nid][0], pos[nid][1], label.get(nid, nid),
                fontsize=8, ha="center", va="bottom", zorder=3)

ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"yED 文件里的实际坐标（{len(pos)} 实体 / {len(edges)} 关系）", fontsize=15)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print("预览图:", OUT)
