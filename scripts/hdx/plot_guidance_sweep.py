#!/usr/bin/env python3
"""Native contacts over time for walk-jump folding at several guidance weights.

Q(t) is the fraction of native C-alpha contacts (pairs within 8 A in the crystal,
|i - j| >= 4) present in a frame, counting a contact as formed below 1.5x its
native distance. One panel per protein, one line per guidance weight (median
over starts), so the weight at which folding is gradual rather than immediate
can be read off.

Usage:
    python scripts/hdx/plot_guidance_sweep.py --hdx_dir $HDX_DIR --out figures/folding/guidance_sweep.png
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

INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#ffffff", "#e6e5e0"
# Guidance weight is an ordered magnitude: one-hue blue ramp (light -> dark), unguided in grey.
WEIGHTS = ["0.001", "0.003", "0.01", "0.02", "0.1"]
RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
PROTEINS = [("1pga", "Protein G B1 (56 aa)"), ("1ubq", "Ubiquitin (76 aa)")]


def native_contacts(ref):
    ca = ref.topology.select("name CA")
    x = ref.xyz[0, ca]
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    i, j = np.triu_indices(len(ca), k=4)
    m = d[i, j] < 0.8
    return i[m], j[m], d[i, j][m]


def q_curve(xtc, ii, jj, d0, stride):
    t = md.load(xtc, top=xtc.replace(".xtc", ".cif"), stride=stride)
    x = t.xyz[:, t.topology.select("name CA")]
    return (np.linalg.norm(x[:, ii] - x[:, jj], axis=-1) < 1.5 * d0).mean(1), stride * np.arange(t.n_frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdx_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--proteins", nargs="+", default=[p for p, _ in PROTEINS], help="PDB ids, one panel each")
    args = parser.parse_args()
    h = Path(args.hdx_dir)

    proteins = [(p, t) for p, t in PROTEINS if p in args.proteins]
    fig, axes = plt.subplots(1, len(proteins), figsize=(6.5 * len(proteins) + 1.5, 4.6), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for ax, (p, title) in zip(axes, proteins):
        ref = md.load(str(h / "structures" / f"{p}.cif"))
        ref = ref.atom_slice(ref.topology.select("protein"))
        ii, jj, d0 = native_contacts(ref)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for cond, color, label, style in ([("unguided", MUTED, "unguided", "--")]
                                          + [(f"guided_w{w}", c, f"weight {w}", "-") for w, c in zip(WEIGHTS, RAMP)]):
            files = sorted(glob.glob(str(h / f"runs_fold_{p}" / cond / "start*" / "boltz_jump_results_*"
                                         / "predictions" / "*" / "*_model_0.xtc")))
            if not files:
                continue
            curves = [q_curve(f, ii, jj, d0, args.stride) for f in files]
            n = min(len(q) for q, _ in curves)
            q = np.median(np.stack([c[0][:n] for c in curves]), axis=0)
            ax.plot(curves[0][1][:n], q, color=color, lw=2, ls=style, label=label)
            first = np.argmax(q >= 0.8) if (q >= 0.8).any() else None
            print(f"{p} {cond:<16} starts {len(files)}  final Q {q[-1]:.2f}  "
                  f"first step with Q >= 0.8: {curves[0][1][first] if first is not None else 'never'}")
        ax.set_title(title, color=INK, fontsize=16, loc="left")
        ax.set_xlabel("walk step", color=INK2, fontsize=16)
        ax.tick_params(colors=INK2, labelsize=15)
    axes[0].set_ylabel("native contacts Q\n(median of 3 starts)", color=INK2, fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    if len(proteins) > 1:
        fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=14, labelcolor=INK2,
                   bbox_to_anchor=(0.5, 1.04))
        fig.suptitle("Folding from random coil: guidance towards the crystal structure at several weights",
                     color=INK, fontsize=18, y=1.1)
    else:
        axes[0].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=14,
                       labelcolor=INK2, title="guidance", title_fontsize=14, alignment="left")
        axes[0].set_title(f"{proteins[0][1]} from random coil,\nguided towards the crystal structure",
                          color=INK, fontsize=17, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
