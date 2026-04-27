"""Render a top-down diagram of a scene from its meta sidecar's `layout` block.

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    python scripts/visionsim/scene_topdown.py data/sim_scenes/tabletop_cliff/
"""

import json
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def render(scene_dir):
    meta_path = os.path.join(scene_dir, "scene.meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No scene.meta.json in {scene_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    if "layout" not in meta:
        raise KeyError(f"{meta_path} has no `layout` block; cannot draw top-down")

    L = meta["layout"]
    table = L["table"]
    room = L["room"]
    fig, ax = plt.subplots(figsize=(8, 9))

    rx0, rx1 = room["x"]
    ry0, ry1 = room["y"]
    ax.add_patch(mpatches.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                    facecolor="#cfcfcf", edgecolor="black", linewidth=1,
                                    label=f"Floor (z={room['z'][0]:.2f})"))

    tx0, tx1 = table["x"]
    ty0, ty1 = table["y"]
    ax.add_patch(mpatches.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0,
                                    facecolor="#d9b380", edgecolor="black", linewidth=1.5,
                                    label=f"Table (z={table['z']:.2f})"))

    for cliff in L["cliffs"]:
        a, b = cliff["range"]
        if cliff["axis"] == "y":
            xs, ys = [a, b], [cliff["value"], cliff["value"]]
        else:
            xs, ys = [cliff["value"], cliff["value"]], [a, b]
        ax.plot(xs, ys, color="red", linewidth=4, linestyle="-",
                label=f"{cliff['name'].capitalize()} cliff ({abs(b - a):.1f} m)")

    for d in L["distractors"]:
        ax.plot(d["x"], d["y"], "o", color=d["color"], markersize=10, markeredgecolor="black")
        ax.annotate(d["name"], (d["x"], d["y"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)

    cam = L["camera_path"]
    sx, sy = cam["start"]
    ex, ey = cam["end"]
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=2.5))
    ax.plot(sx, sy, "s", color="black", markersize=12)
    ax.annotate(f"CAM start\n({sx:.2f}, {sy:.2f})", (sx, sy),
                textcoords="offset points", xytext=(-95, -5), fontsize=9, fontweight="bold")
    ax.annotate(f"CAM end\n({ex:.2f}, {ey:.2f})", (ex, ey),
                textcoords="offset points", xytext=(-95, 5), fontsize=9, fontweight="bold")

    ax.plot(0, 0, "+", color="black", markersize=15, markeredgewidth=2)
    ax.annotate("world origin", (0, 0), textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlim(rx0 - 1.2, rx1 + 0.5)
    ax.set_ylim(ry0 - 1.2, ry1 + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m, +Y = forward)")
    ax.set_title(f"Top-down: {meta.get('scene_name', scene_dir)}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(scene_dir, "scene_topdown.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Top-down saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <scene_dir>", file=sys.stderr)
        sys.exit(1)
    render(sys.argv[1])
