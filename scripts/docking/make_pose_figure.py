#!/usr/bin/env python3
"""Slide figure for one pose-refinement example: co-folded seed vs walk-jump.

Renders the export_pose_figure.py output with render_pose_figure.pml (two panels,
same camera) and labels each panel with its ligand RMSD to the crystal pose.

Usage:
    python scripts/docking/make_pose_figure.py --pose_dir $DOCKING_DIR/figure_exports/8DW5_FQ7 \
        --title "PoseBusters 8DW5 (ligand FQ7)" --pymol $(which pymol) --out figures/docking/posebusters_8DW5_poses.png
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

INK, INK2 = "#0b0b0b", "#52514e"
CRYSTAL, SEED, WALKJUMP = "#808080", "#eb6834", "#2a78d6"


def content_box(img, pad=30):
    """Bounding box of the non-white pixels, padded."""
    x0, y0, x1, y1 = Image.eval(img.convert("L"), lambda v: 255 if v < 247 else 0).getbbox()
    return max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, img.width), min(y1 + pad, img.height)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose_dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--pymol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    info = json.load(open(Path(args.pose_dir) / "poses.json"))
    pml = Path(__file__).with_name("render_pose_figure.pml")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([args.pymol, "-cq", str(pml), "--", args.pose_dir, f"{tmp}/panel"], check=True)
        panels = [Image.open(f"{tmp}/panel_{k}.png").convert("RGB") for k in ("seed", "walkjump")]
    # one crop for both panels (union of their content) so the camera stays identical
    boxes = [content_box(p) for p in panels]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    panels = [p.crop(box) for p in panels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), facecolor="white")
    titles = [(f"Co-folded seed (Boltz-2): {info['seed_lig_rmsd']:.1f} Å", SEED),
              (f"Boltz-Jump: {info['walkjump_rmsd']:.1f} Å", WALKJUMP)]
    for ax, img, (title, color) in zip(axes, panels, titles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, color=INK, fontsize=18)
    handles = [Line2D([], [], color=CRYSTAL, lw=6, alpha=0.65, label="crystal pose"),
               Line2D([], [], color=SEED, lw=6, label="co-folded seed"),
               Line2D([], [], color=WALKJUMP, lw=6, label="Boltz-Jump (best of 200 frames)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=15, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(args.title, color=INK, fontsize=20, y=0.99)
    fig.text(0.5, -0.07, f"Ligand RMSD to the crystal pose after pocket superposition. Walk-jump from this seed, "
             f"σ = {info['sigma']} Å, frame {info['frame']} of {info['n_frames']}; protein (cartoon) from that frame.",
             ha="center", color=INK2, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
