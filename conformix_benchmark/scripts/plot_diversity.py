#!/usr/bin/env python3
"""Figures for the walk-jump exploration work.

1. bias_height_sweep.png: distance from the start and backbone integrity over a
   30-step walk, per bias height, against the unbiased walk (same random seed).
2. conformix_matched_compute.png: ConforMix coverage@50% for Boltz-2 diffusion
   (100 and 10 samples) and walk-jump, on the same targets.

Usage:
    python conformix_benchmark/scripts/plot_diversity.py --sweep_dir logs/bias_sweep_out \
        --baseline_dir /tmp/.../none --out_dir figures/diversity
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent

# Reference palette (dataviz skill, references/palette.md), light mode.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3: valid for all pairs
ORDINAL_BLUE = ["#86b6ef", "#2a78d6", "#104281"]  # steps 250, 450, 650
HEIGHTS = ["0.005", "0.02", "0.08"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def walk_stats(root):
    p = glob.glob(f"{root}/boltz_jump_results_*/predictions/*/*_model_0.xtc")[0]
    t = md.load(p, top=p.replace(".xtc", ".cif"))
    ca = t.top.select("name CA")
    rmsd = 10 * md.rmsd(t, t, 0, atom_indices=ca)
    xyz = 10 * t.xyz[:, ca]
    gap = np.abs(np.linalg.norm(np.diff(xyz, axis=1), axis=-1) - 3.8).max(1)
    return rmsd, gap


def sweep_figure(sweep_dir, baseline_dir, out):
    base_rmsd, base_gap = walk_stats(baseline_dir)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, facecolor=SURFACE)
    for col, (mode, title) in enumerate([("seed", "Seed repulsion"), ("metad", "Metadynamics")]):
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            style(ax)
        steps = np.arange(len(base_rmsd))
        top.plot(steps, base_rmsd, color=MUTED, lw=2, ls="--", label="no bias")
        bottom.plot(steps, base_gap, color=MUTED, lw=2, ls="--", label="no bias")
        for h, c in zip(HEIGHTS, ORDINAL_BLUE):
            rmsd, gap = walk_stats(Path(sweep_dir) / f"{mode}_{h}")
            top.plot(steps, rmsd, color=c, lw=2, label=f"height {h}")
            bottom.plot(steps, gap, color=c, lw=2, label=f"height {h}")
            if h == HEIGHTS[-1]:  # the others track the baseline; the legend names them
                top.annotate(f"height {h}", (steps[-1], rmsd[-1]), xytext=(4, 0), textcoords="offset points",
                             color=INK2, fontsize=8, va="center")
        top.set_title(title, color=INK, fontsize=11, loc="left")
        bottom.set_xlabel("walk step", color=INK2, fontsize=9)
        bottom.set_ylim(0, 0.3)
    axes[0, 0].set_ylabel("C$\\alpha$ RMSD from start (Å)", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("max |C$\\alpha$–C$\\alpha$ − 3.8 Å| (Å)", color=INK2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Bias height sweep: 30 steps, 110-residue fold-switching protein (fs-4qhf)",
                 color=INK, fontsize=12, y=1.05)
    fig.text(0.5, -0.02, "Heights of 0.5–2 (not shown) broke the backbone: C$\\alpha$ gaps of 30+ Å.",
             ha="center", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def coverage_figure(out):
    subsets = [("domainmotion", "Domain motion"), ("crypticpockets", "Cryptic pockets"),
               ("foldswitching", "Fold switching"), ("membranetransporters", "Membrane transporters")]
    methods = [("diffusion_100", "Boltz-2 diffusion, 100 samples"),
               ("diffusion_10", "Boltz-2 diffusion, 10 samples"),
               ("msa_boltz2init_d2_1000", "Walk-jump, 1000 steps")]
    rows = []
    for key, label in subsets:
        frames = [pd.read_csv(HERE / "eval_kabsch" / m / f"{key}.csv")[["system_id", "best_frac", "alt_frac"]]
                  .rename(columns={"best_frac": f"b_{m}", "alt_frac": f"a_{m}"}) for m, _ in methods]
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on="system_id")
        for m, mlabel in methods:
            rows.append({"subset": label, "n": len(merged), "method": mlabel,
                         "best": (merged[f"b_{m}"] <= 0.5).mean(), "alt": (merged[f"a_{m}"] <= 0.5).mean()})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    ylabels = [f"{s} (n={df[df.subset == s].n.iloc[0]})" for _, s in subsets]
    y = np.arange(len(subsets))[::-1]
    offsets = [0.2, 0.0, -0.2]
    for ax, metric, title in [(axes[0], "best", "Best-matched state"), (axes[1], "alt", "Alternate state")]:
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for (_, mlabel), c, off in zip(methods, CATEGORICAL, offsets):
            v = [df[(df.subset == s) & (df.method == mlabel)][metric].iloc[0] for _, s in subsets]
            ax.scatter(v, y + off, s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=3, label=mlabel)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("coverage@50% (fraction of targets)", color=INK2, fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ylabels, color=INK2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("ConforMix: walk-jump vs Boltz-2 diffusion on the same targets", color=INK, fontsize=12, y=1.1)
    fig.text(0.5, -0.06, "Walk-jump (1000 steps) costs about as much as 10 diffusion samples. "
             "Alternate-state differences at matched compute are within noise (4 vs 2 and 9 vs 5 targets).",
             ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    df.to_csv(Path(out).with_suffix(".csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--out_dir", default="figures/diversity")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweep_figure(args.sweep_dir, args.baseline_dir, out / "bias_height_sweep.png")
    coverage_figure(out / "conformix_matched_compute.png")
    print(f"wrote {out}/bias_height_sweep.png, conformix_matched_compute.png (+ .csv)")


if __name__ == "__main__":
    main()
