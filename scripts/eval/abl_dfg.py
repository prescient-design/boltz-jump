#!/usr/bin/env python3
"""DFG-motif distances for Abl kinase ensembles (Modi & Dunbrack, PNAS 2019).

    D1 = |CA(aC-Glu + 4) - CZ(DFG-Phe)|      M309 CA  - F401 CZ
    D2 = |CA(beta3-Lys)  - CZ(DFG-Phe)|      K290 CA  - F401 CZ

(Abl 1b numbering, as in 6XR6 / 6XR7.) DFG-in has the Phe packed under the aC
helix (small D1, large D2); DFG-out flips it towards the beta3 Lys (large D1,
small D2). Residues are taken by position in the construct, which starts at
Ser248, so the Boltz numbering 1..287 maps onto it directly.

Writes the per-frame distances (CSV) and a figure: D1 vs D2 for every sample,
and D1 along each walk-jump chain.

Usage:
    python scripts/eval/abl_dfg.py --runs $RESULTS_DIR/abl_kinase_states --out figures/abl_kinase/abl_dfg.png
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.labelsize": 16})
import mdtraj as md
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIRST_RESSEQ = 248
GLU4, LYS, PHE = 309, 290, 401
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#ffffff", "#e6e5e0"
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
REFS = [("6XR6", "active (6XR6)"), ("6XR7", "inactive I1 (6XR7)")]
CHAINS = [("6XR6", "Boltz-Jump from active", BLUE), ("6XR7", "Boltz-Jump from inactive", AQUA)]


def atom(top, resseq, name):
    """Atom index by construct position (works for crystal and Boltz numbering alike)."""
    res = [r for r in top.residues if r.is_protein][resseq - FIRST_RESSEQ]
    return next(a.index for a in res.atoms if a.name == name)


def dfg(t):
    pairs = [[atom(t.top, GLU4, "CA"), atom(t.top, PHE, "CZ")], [atom(t.top, LYS, "CA"), atom(t.top, PHE, "CZ")]]
    d = md.compute_distances(t, pairs) * 10
    return d[:, 0], d[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="output directory of scripts/slurm/abl_kinase_state_preference.sh")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = Path(args.runs)

    rows = []
    for code, label in REFS:
        d1, d2 = dfg(md.load(str(ROOT / "structures" / f"{code}.cif")))
        rows.append({"set": label, "frame": 0, "D1": d1[0], "D2": d2[0]})
    for code, label, _ in CHAINS:
        pred = runs / f"boltz_jump_results_s=2.0_d=1.0_f=0.1_abl_kinase_from_{code}" / "predictions" \
            / f"abl_kinase_from_{code}" / f"abl_kinase_from_{code}_model_0"
        d1, d2 = dfg(md.load(f"{pred}.xtc", top=f"{pred}.cif"))
        rows += [{"set": label, "frame": i, "D1": a, "D2": b} for i, (a, b) in enumerate(zip(d1, d2))]
    cifs = sorted(glob.glob(str(runs / "boltz_results_*_abl_kinase" / "predictions" / "abl_kinase" / "*_model_*.cif")))
    for i, f in enumerate(cifs):
        d1, d2 = dfg(md.load(f))
        rows.append({"set": "Boltz-2 diffusion", "frame": i, "D1": d1[0], "D2": d2[0]})
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    for s, g in df.groupby("set", sort=False):
        print(f"{s:<28} n={len(g):<5} D1 median {g.D1.median():5.1f} [{g.D1.min():.1f}, {g.D1.max():.1f}]   "
              f"D2 median {g.D2.median():5.1f} [{g.D2.min():.1f}, {g.D2.max():.1f}]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK2)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    ax = axes[0]
    g = df[df.set == "Boltz-2 diffusion"]
    ax.scatter(g.D1, g.D2, s=40, color=ORANGE, alpha=0.8, edgecolor=SURFACE, lw=0.8, label=f"Boltz-2 diffusion ({len(g)})",
               zorder=4)
    for _, label, color in CHAINS:
        g = df[df.set == label]
        ax.scatter(g.D1, g.D2, s=10, color=color, alpha=0.35, lw=0, label=f"{label} ({len(g)} frames)", zorder=3)
    for (code, label), marker in zip(REFS, ("*", "P")):
        g = df[df.set == label]
        ax.scatter(g.D1, g.D2, s=380, marker=marker, color=INK, edgecolor=SURFACE, lw=1.5, label=label, zorder=5)
    ax.set_xlabel("D1: αC-Glu+4 Cα – DFG-Phe Cζ (Å)", color=INK2)
    ax.set_ylabel("D2: β3-Lys Cα – DFG-Phe Cζ (Å)", color=INK2)
    ax.set_title("DFG conformation of every sample", color=INK, fontsize=17, loc="left")

    ax = axes[1]
    for _, label, color in CHAINS:
        g = df[df.set == label]
        ax.plot(g.frame, g.D1, color=color, lw=1.4, label=label)
    for (code, label), ls in zip(REFS, ("--", ":")):
        y = df[df.set == label].D1.iloc[0]
        ax.axhline(y, color=INK, lw=1.4, ls=ls)
        ax.text(1.0, y, f" {label.split(' (')[0]}\n {code}", transform=ax.get_yaxis_transform(), va="center",
                color=INK2, fontsize=13)
    ax.set_xlabel("walk step", color=INK2)
    ax.set_ylabel("D1 (Å)", color=INK2)
    ax.set_title("D1 along each Boltz-Jump chain", color=INK, fontsize=17, loc="left")

    h0, l0 = axes[0].get_legend_handles_labels()
    leg = fig.legend(h0, l0, loc="upper center", ncol=3, frameon=False, fontsize=13.5, labelcolor=INK2,
                     bbox_to_anchor=(0.5, 1.1), markerscale=1)
    for hnd in leg.legend_handles:
        hnd.set_alpha(1)
    fig.text(0.5, -0.04, "Abl kinase domain (1b numbering): D1 = M309 Cα – F401 Cζ, D2 = K290 Cα – F401 Cζ "
             "(Modi & Dunbrack 2019). σ = 2 Å, 1000 steps per chain.", ha="center", color=INK2, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
