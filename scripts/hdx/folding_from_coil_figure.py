#!/usr/bin/env python3
"""Slide figure: unguided walk-jump from random coil (protein G, ubiquitin), best of 3 starts.

Composes the PyMOL renders <pdb>_{start,bestQ,final,native}.png in --dir into
--dir/folding_from_coil.png. Labels (Q, Ca RMSD) are from the runs in
figures/talk/talk_notes.md section 3.

Usage:
    python scripts/hdx/folding_from_coil_figure.py --dir figures/folding
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"
ROWS = [("1pga", "Protein G B1\n(56 aa)", ["start\nQ 0.03, 15.6 Å", "most native frame\nQ 0.46, 11.1 Å",
                                            "final (step 3000)\nQ 0.35, 9.8 Å", "native (1PGA)"]),
        ("1ubq", "Ubiquitin\n(76 aa)", ["start\nQ 0.08, 25.1 Å", "most native frame\nQ 0.39, 13.8 Å",
                                         "final (step 3000)\nQ 0.31, 13.1 Å", "native (1UBQ)"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="figures/folding")
    args = parser.parse_args()
    d = Path(args.dir)
    fig, axes = plt.subplots(2, 4, figsize=(13, 8.4), facecolor=SURFACE)
    for r, (pdb, title, labels) in enumerate(ROWS):
        for c, (tag, label) in enumerate(zip(["start", "bestQ", "final", "native"], labels)):
            ax = axes[r, c]
            ax.imshow(mpimg.imread(d / f"{pdb}_{tag}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(label, color=INK2, fontsize=15)
        axes[r, 0].set_ylabel(title, color=INK, fontsize=17)
    fig.suptitle("Unguided walk-jump from random coil (σ = 2 Å, 3000 steps), best of 3 starts", color=INK, fontsize=18)
    fig.text(0.5, 0.01, "Blue N-terminus to red C-terminus, superposed on the native.\n"
             "Q: fraction of native Cα contacts; Å: Cα RMSD to native.", ha="center", color=INK2, fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), h_pad=3.5)
    fig.savefig(d / "folding_from_coil.png", dpi=150, facecolor=SURFACE)
    print(f"-> {d / 'folding_from_coil.png'}")


if __name__ == "__main__":
    main()
