"""Render a top-down diagram of a scene from its meta sidecar's `layout` block.

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    python scripts/visionsim/scene_topdown.py data/sim_dataset/tabletop_cliff/scene/
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

    for shelf in L.get("shelves", []):
        sx0, sx1 = shelf["x"]
        sy0, sy1 = shelf["y"]
        ax.add_patch(mpatches.Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0,
                                        facecolor="#7a7a82", edgecolor="black",
                                        linewidth=1, alpha=0.85))
        ax.annotate(shelf["name"], ((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0),
                    fontsize=7, ha="center", va="center", color="white")

    for i, light in enumerate(L.get("lights", [])):
        lx0, lx1 = light["x"]
        ly0, ly1 = light["y"]
        label = (f"Lights ({len(L['lights'])} x {light.get('energy_w', '?')} W "
                 f"@ z={light['z']:.2f})") if i == 0 else None
        ax.add_patch(mpatches.Rectangle((lx0, ly0), lx1 - lx0, ly1 - ly0,
                                        facecolor="#fff0a8", edgecolor="#a08800",
                                        linewidth=0.8, alpha=0.85, label=label))

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
    # Push CAM labels to the side of the room with more clearance: if the path
    # is on the right half of the room, label to the right; otherwise to the left.
    rxmid = (rx0 + rx1) / 2.0
    cam_side, cam_ha = (15, "left") if sx >= rxmid else (-15, "right")
    ax.annotate(f"CAM start\n({sx:.2f}, {sy:.2f})", (sx, sy),
                textcoords="offset points", xytext=(cam_side, -18), fontsize=9,
                fontweight="bold", ha=cam_ha, va="top")
    ax.annotate(f"CAM end\n({ex:.2f}, {ey:.2f})", (ex, ey),
                textcoords="offset points", xytext=(cam_side, 18), fontsize=9,
                fontweight="bold", ha=cam_ha, va="bottom")

    ax.plot(0, 0, "+", color="black", markersize=15, markeredgewidth=2)
    # Origin label drops to lower-left so it can't collide with a CAM start/end
    # label that lands near the world origin.
    ax.annotate("world origin", (0, 0), textcoords="offset points",
                xytext=(-10, -10), fontsize=8, ha="right", va="top")

    ax.set_xlim(rx0 - 1.2, rx1 + 0.5)
    ax.set_ylim(ry0 - 1.2, ry1 + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m, +Y = forward)")
    ax.set_title(f"Top-down: {meta.get('scene_name', scene_dir)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=9)
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
