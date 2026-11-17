#!/usr/bin/env python3
"""Slide figures for the Boltz-Jump talk, from the verified result tables.

    figures/talk/1_ensembles.png    ATLAS + mdCATH ensemble metrics (paper's main tables)
    figures/talk/1b_noised_seed.png ablation: walk-jump vs noising the seed once and denoising it
    figures/talk/1c_walk_denoise.png ablation: one-step jump vs N-step reverse diffusion (walk-denoise)
    figures/talk/1d_convergence_nfe.png the same against network function evaluations
    figures/talk/1d_convergence.png RMWD to the MD reference vs number of samples
                                    (from scripts/slurm/convergence.sh; _all = every heavy atom)
    figures/talk/2_ligand.png       Runs N' Poses: rescue vs seed distance (oracle)
    figures/talk/4_hdx.png          Start2Fold: STRONG > MEDIUM AUC gain over the static structure, per protein

Folding figures (3): figures/talk/3_guidance_sweep.png is written by
scripts/hdx/plot_guidance_sweep.py; the others are in figures/folding/.

Usage:
    python scripts/make_talk_figures.py --hdx_dir $HDX_DIR --docking_dir $DOCKING_DIR
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.labelsize": 16})
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#ffffff", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4, in order
LIGHT_BLUE = "#86b6ef"  # Boltz-Jump from a Boltz-2 seed: same method as BLUE, lighter step


def caption(fig, y, text, **kw):
    """Figure caption below the panels, wrapped so it doesn't widen the canvas."""
    kw.setdefault("ha", "center")
    fig.text(0.5, y, textwrap.fill(text, 110), va="top", **kw)


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=15)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# Shared by 1, 1b, 1c: the contact metrics, then structure quality and diversity.
METRICS = [("Weak contacts J", "Weak\ncontacts"), ("Transient contacts J", "Transient\ncontacts"),
           ("Exposed residue J", "Exposed\nresidues"), ("lDDT Precision", "lDDT\nprecision"),
           ("lDDT Diversity", "lDDT\ndiversity")]


def ensembles(out):
    models = [("Boltz-2 Jump (MSA, δ=1)", "Boltz-Jump (crystal seed)", BLUE),
              ("Boltz-2 Jump (MSA, no init, δ=1)", "Boltz-Jump (Boltz-2 seed)", LIGHT_BLUE),
              ("Boltz-2 Full MSA", "Boltz-2 diffusion", ORANGE),
              ("AlphaFlow MD Base", "AlphaFlow-MD", MUTED),
              ("BioEmu", "BioEmu", "#c3c2b7")]
    metrics = METRICS
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (ds, title) in zip(axes, [("atlas", "ATLAS (40 proteins)"), ("mdcath", "mdCATH, 320 K (40 proteins)")]):
        d = pd.read_csv(ROOT / "results_plots_nov2026" / ds / "summary_table.csv", index_col=0)
        style(ax)
        x = np.arange(len(metrics))
        w = 0.16
        for k, (row, label, color) in enumerate(models):
            vals = [d.loc[row, m] for m, _ in metrics]
            ax.bar(x + (k - (len(models) - 1) / 2) * w, vals, w * 0.92, color=color, label=label, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([m for _, m in metrics], color=INK2)
        ax.set_title(title, color=INK, fontsize=18, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("score (higher is better)", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=15, labelcolor=INK2, bbox_to_anchor=(0.5, 1.13))
    caption(fig, -0.02, "Boltz-Jump: same pre-trained Boltz-2, no training; 5-10x faster than diffusion; seeded from the crystal or from Boltz-2's own prediction. "
             "AlphaFlow-MD and BioEmu are trained on MD.", ha="center", color=INK2, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def noised_seed(out):
    # The three noised-seed baselines are one family: shades of green, dark to light.
    models = [("Boltz-2 Jump (MSA, δ=1)", "Boltz-Jump (walk + jump)", BLUE),
              ("Noised Seed SDE (fixed)", "noise seed, denoise: SDE", "#0b6b47"),
              ("Noised Seed ODE (fixed)", "noise seed, denoise: ODE", "#1baf7a"),
              ("Noised Seed (1-step)", "noise seed, denoise: 1 step", "#8fdcb8")]
    metrics = METRICS
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE)
    for ax, (ds, title) in zip(axes, [("atlas", "ATLAS"), ("mdcath", "mdCATH, 320 K")]):
        d = pd.read_csv(ROOT / "results_plots_nov2026" / ds / "summary_table.csv", index_col=0)
        style(ax)
        x = np.arange(len(metrics))
        w = 0.19
        for k, (row, label, color) in enumerate(models):
            for i, (m, _) in enumerate(metrics):
                v = d.loc[row, m]
                pos = x[i] + (k - 1.5) * w
                if np.isnan(v):  # exposed residues are undefined for the 1-step baseline
                    ax.text(pos, 0.01, "n/a", ha="center", va="bottom", color=MUTED, fontsize=11, rotation=90)
                    continue
                ax.bar(pos, v, w * 0.92, color=color, label=label if i == 0 else None, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([m for _, m in metrics], color=INK2)
        ax.set_title(title, color=INK, fontsize=18, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("score (higher is better)", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=15, labelcolor=INK2, bbox_to_anchor=(0.5, 1.13))
    caption(fig, -0.02, "Same seed, same noise level (sigma = 2 A). Denoising a noised seed gives slightly better "
             "single structures (lDDT precision 0.88 vs 0.86 on ATLAS) but 3-5x less diversity.",
             ha="center", color=INK2, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def walk_denoise(out):
    # Boltz-Jump keeps its colour from 1 and 1b; N = number of reverse-diffusion steps per saved frame.
    models = [("Boltz-2 Jump (MSA, δ=1)", "Boltz-Jump (1-step jump)", BLUE),
              ("Boltz-2 Walk-Denoise (MSA, δ=1, N=10)", "walk-denoise, N = 10 denoising steps", YELLOW)]
    metrics = METRICS
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    for ax, (ds, title) in zip(axes, [("atlas", "ATLAS"), ("mdcath", "mdCATH, 320 K")]):
        d = pd.read_csv(ROOT / "results_plots_nov2026" / ds / "summary_table.csv", index_col=0)
        style(ax)
        x = np.arange(len(metrics))
        w = 0.34
        for k, (row, label, color) in enumerate(models):
            ax.bar(x + (k - 0.5) * w, [d.loc[row, m] for m, _ in metrics], w * 0.92, color=color, label=label, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([m for _, m in metrics], color=INK2, fontsize=15)
        ax.set_title(title, color=INK, fontsize=18, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("score (higher is better)", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=15, labelcolor=INK2, bbox_to_anchor=(0.5, 1.07))
    caption(fig, -0.04, "Same walk; each saved frame denoised by one Tweedie step (Boltz-Jump) or a 10-step reverse "
            "diffusion from sigma = 2 A. 10 steps: slightly better single structures (lDDT precision +0.01), ~30% less "
            "diversity, weaker contact metrics (exposed residues mixed).", ha="center", color=INK2, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def convergence(out, subdir="convergence_seeded"):
    models = [("boltz2_jump_msa_d1", "Boltz-Jump (crystal seed)", BLUE),
              ("boltz2_jump_msa_no_init_d1", "Boltz-Jump (Boltz-2 seed)", LIGHT_BLUE),
              ("boltz2_full_msa", "Boltz-2 diffusion", ORANGE),
              ("alphaflow_md_base", "AlphaFlow-MD", MUTED), ("bioemu", "BioEmu", "#c3c2b7")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, (ds, title) in zip(axes, [("atlas_test", "ATLAS"), ("mdcath", "mdCATH, 320 K")]):
        d = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "results_analysis" / subdir / ds).glob("*.csv")))
        # Only targets every model covers, so the medians compare the same proteins.
        complete = d.groupby("target")["model"].nunique()
        d = d[d.target.isin(complete[complete == d.model.nunique()].index)]
        style(ax, grid="both")
        for model, label, color in models:
            if model not in set(d.model):
                continue
            q = d[d.model == model].groupby("n")["rmwd"].median()
            ax.plot(q.index, q.values, color=color, lw=2, marker="o", ms=8, label=label, zorder=3)
        w = d[d.n == d.n.max()].pivot(index="target", columns="model", values="rmwd")
        wins = [f"{seed} seed {(w[m] < w['boltz2_full_msa']).sum()}/{len(w)}"
                for m, seed in (("boltz2_jump_msa_d1", "crystal"), ("boltz2_jump_msa_no_init_d1", "Boltz-2")) if m in w]
        ax.text(0.5, -0.24, f"N = {d.n.max()}, Boltz-Jump beats diffusion:\n" + ", ".join(wins), transform=ax.transAxes,
                ha="center", va="top", color=INK2, fontsize=13)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
        ax.set_xlabel("number of samples", color=INK2)
        ax.set_title(f"{title} ({d.target.nunique()} proteins)", color=INK, fontsize=18, loc="left")
    axes[0].set_ylabel("RMWD to MD (Å, lower is better)", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=15, labelcolor=INK2, bbox_to_anchor=(0.5, 1.13))
    caption(fig, -0.09, "Median over proteins. Walk-jump: first N frames of one chain; "
             "others: first N independent samples. Heavy atoms present in the walk-jump seed.", ha="center", color=INK2, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def convergence_nfe(out, subdir="convergence_seeded"):
    # Network function evaluations (score-model calls; the trunk runs once for every method and is excluded).
    # Diffusion: 200 sampling steps per sample. Walk-jump: one call per BAOAB step, one frame saved per step
    # (the jump reuses that step's denoised structure); the Boltz-2 seed adds one 200-step diffusion sample.
    models = [("boltz2_jump_msa_d1", "Boltz-Jump (crystal seed)", BLUE, lambda n: n),
              ("boltz2_jump_msa_no_init_d1", "Boltz-Jump (Boltz-2 seed)", LIGHT_BLUE, lambda n: 200 + n),
              ("boltz2_full_msa", "Boltz-2 diffusion", ORANGE, lambda n: 200 * n)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, (ds, title) in zip(axes, [("atlas_test", "ATLAS"), ("mdcath", "mdCATH, 320 K")]):
        d = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "results_analysis" / subdir / ds).glob("*.csv")))
        complete = d.groupby("target")["model"].nunique()
        d = d[d.target.isin(complete[complete == d.model.nunique()].index)]
        style(ax, grid="both")
        for model, label, color, nfe in models:
            q = d[d.model == model].groupby("n")["rmwd"].median()
            ax.plot([nfe(n) for n in q.index], q.values, color=color, lw=2, marker="o", ms=8, label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
        ax.set_xlabel("network function evaluations", color=INK2)
        ax.set_title(f"{title} ({d.target.nunique()} proteins)", color=INK, fontsize=18, loc="left")
    axes[0].set_ylabel("RMWD to MD (Å, lower is better)", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=15, labelcolor=INK2, bbox_to_anchor=(0.5, 1.07))
    caption(fig, -0.03, "Points: 2-100 samples. Diffusion: 200 NFE/sample; "
             "walk-jump: 1/frame (+200 for a Boltz-2 seed).", ha="center", color=INK2, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def ligand(out, docking_dir):
    t = pd.read_csv(Path(docking_dir) / "runs_n_poses" / "analysis" / "targets.csv")
    t = t[t.sigma == 4.0].dropna(subset=["best_ideal_rmsd"])
    seed, best = t.best_start_lig_rmsd.values, t.best_ideal_rmsd.values
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style(ax)
    bands = [("< 4 Å", seed < 4), ("4-7 Å", (seed >= 4) & (seed < 7)), ("> 7 Å", seed >= 7)]
    for k, (label, m) in enumerate(bands):
        n, ok = int(m.sum()), int((best[m] < 2).sum())
        ax.bar(k, ok / n, 0.6, color=BLUE, zorder=2)
        ax.text(k, ok / n + 0.02, f"{ok}/{n}", ha="center", color=INK2, fontsize=16)
    ax.set_xticks(range(3))
    ax.set_xticklabels([b for b, _ in bands], color=INK2)
    ax.set_xlabel("distance of the best co-folded seed\nfrom the crystal pose", color=INK2)
    ax.set_ylabel("fraction rescued (< 2 Å)", color=INK2)
    ax.set_ylim(0, 0.8)
    ax.set_title(f"Boltz-Jump relaxes poses locally: {int((best < 2).sum())}/{len(best)} rescued",
                 color=INK, fontsize=18, loc="left")
    caption(fig, -0.03, f"Runs N' Poses: all {len(best)} single-ligand co-folding failures (right pocket, wrong pose). "
            "σ = 4 Å, 5 seeds × 200 frames; oracle (best pose chosen with the crystal).",
            ha="center", color=INK2, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def hdx(out, hdx_dir):
    s2f = Path(hdx_dir) / "start2fold"
    series = []
    for name, label, color in [("native", "Boltz-Jump", BLUE), ("repel_h0.15", "Boltz-Jump +\nRMSD diversity", AQUA)]:
        d = pd.read_csv(s2f / f"benchmark_{name}_vs_static.csv")
        # STRONG > MEDIUM only (the pair nearly every protein has); entries of one protein averaged
        per = d.dropna(subset=["strong_vs_medium_gain"]).groupby("uniprot")["strong_vs_medium_gain"].mean()
        series.append((label, color, per))
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    style(ax, grid="x")
    for k, (label, color, per) in enumerate(series):
        v = np.sort(per.values)
        y = np.full(len(v), k) + np.linspace(-0.22, 0.22, len(v))
        ax.scatter(v, y, s=80, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
        wins = int((v > 0).sum())
        ax.text(0.33, k, f"better on {wins}/{len(v)}\nmean {v.mean():+.3f}", va="center", color=INK2, fontsize=15)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in series], color=INK2)
    ax.set_xlim(-0.3, 0.45)
    ax.set_title(f"Native-state HDX protection (Start2Fold, {len(series[0][2])} proteins)",
                 color=INK, fontsize=18, loc="left")
    ax.set_xlabel("AUC gain over the static crystal structure (per protein)", color=INK2)
    caption(fig, -0.05, "AUC: how often a STRONG-protection amide is predicted more protected than a MEDIUM one "
            "(Best-Vendruscolo ln P, averaged over the ensemble); 0.5 = chance.", ha="center", color=INK2, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdx_dir", required=True)
    parser.add_argument("--docking_dir", required=True)
    args = parser.parse_args()
    out = ROOT / "figures" / "talk"
    out.mkdir(parents=True, exist_ok=True)
    ensembles(out / "1_ensembles.png")
    noised_seed(out / "1b_noised_seed.png")
    walk_denoise(out / "1c_walk_denoise.png")
    convergence(out / "1d_convergence.png")
    convergence(out / "1d_convergence_all_atoms.png", "convergence")
    convergence_nfe(out / "1d_convergence_nfe.png")
    ligand(out / "2_ligand.png", args.docking_dir)
    hdx(out / "4_hdx.png", args.hdx_dir)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
