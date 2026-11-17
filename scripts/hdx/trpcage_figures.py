#!/usr/bin/env python3
"""Trp-cage folding (unguided) and unfolding (guided) figures for slides.

Inputs are walk-jump runs of configs/fast_folding/trpcage_unfolded.yaml and
trpcage_folded_to_unfolded.yaml (msa: empty, absolute structure paths) with
sigma = 2, delta = 2, friction 0.1, 1000 steps, seeds 0-3; unfolding adds
--structure_guidance (pull towards final_structure). Run via
scripts/slurm/conformix_diversity.sh with a task list, outputs in <runs>/{folding,unfolding}/seed<k>/.

Writes to --out_dir:
    trpcage_{folding,unfolding}_frame_<i>.png   snapshots (frames 0, 200, ..., 800, 999)
    trpcage_{folding,unfolding}_strip.png       the snapshots side by side
    trpcage_{folding,unfolding}.gif             movie (every 10th frame)
    trpcage_rmsd.png                            Ca RMSD to native vs step, all seeds
    trpcage_{folding,unfolding}_frames.pdb      the rendered frames, superposed on the native

Usage:
    python scripts/hdx/trpcage_figures.py --runs $TRPCAGE_RUNS \
        --pymol $(which pymol) --out_dir figures/folding/trpcage
"""

import argparse
import glob
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.labelsize": 16})
import mdtraj
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "structures" / "trpcage_folded.pdb"
FRAMES = [0, 200, 400, 600, 800, 999]
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#ffffff", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"

MOVIE_PML = """
python
import sys
from pymol import cmd
pdb, ref, prefix = sys.argv[-3], sys.argv[-2], sys.argv[-1]
cmd.load(ref, "ref")
cmd.load(pdb, "mol")
cmd.dss("mol", state=0)
cmd.hide("everything")
cmd.show("cartoon", "mol")
cmd.spectrum("resi", "rainbow", "mol and name CA")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("ray_shadows", 0)
cmd.orient("ref")
cmd.zoom("mol", 2, state=0)
cmd.viewport(500, 500)
cmd.set("ray_trace_frames", 1)
cmd.mpng(prefix, width=500, height=500)
python end
"""


def load(path, native):
    t = mdtraj.load(path, top=path.replace(".xtc", ".cif"))
    ca, nca = t.top.select("name CA"), native.top.select("name CA")
    t.superpose(native, atom_indices=ca, ref_atom_indices=nca)
    rmsd = mdtraj.rmsd(t.atom_slice(ca), native.atom_slice(nca)) * 10
    for a in t.top.atoms:
        a.serial = None
    return t, rmsd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--pymol", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--folding_seed", type=int, default=1)
    parser.add_argument("--unfolding_seed", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    native = mdtraj.load(str(NATIVE))
    render = ROOT / "scripts" / "hdx" / "render_folding.pml"

    traces = {}
    for kind, shown in [("folding", args.folding_seed), ("unfolding", args.unfolding_seed)]:
        traces[kind] = {}
        for path in sorted(glob.glob(f"{args.runs}/{kind}/seed*/boltz_jump_results_*/predictions/*/*_model_0.xtc")):
            seed = int(path.split(f"/{kind}/seed")[1].split("/")[0])
            t, rmsd = load(path, native)
            traces[kind][seed] = rmsd
            if seed != shown:
                continue
            t[FRAMES].save_pdb(str(out / f"trpcage_{kind}_frames.pdb"))
            panels = []
            with tempfile.TemporaryDirectory() as tmp:
                for i in FRAMES:
                    pdb = f"{tmp}/frame_{i}.pdb"
                    t[i].save_pdb(pdb)
                    png = out / f"trpcage_{kind}_frame_{i:04d}.png"
                    subprocess.run([args.pymol, "-cq", str(render), "--", pdb, str(NATIVE), str(png)], check=True)
                    panels.append((i, rmsd[i], png))
                movie = f"{tmp}/movie.pdb"
                t[::10].save_pdb(movie)
                pml = f"{tmp}/movie.pml"
                Path(pml).write_text(MOVIE_PML)
                subprocess.run([args.pymol, "-cq", pml, "--", movie, str(NATIVE), f"{tmp}/m_"], check=True)
                frames = [Image.open(f) for f in sorted(glob.glob(f"{tmp}/m_*.png"))]
                frames[0].save(out / f"trpcage_{kind}.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)

            fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0), facecolor="white")
            for ax, (i, r, png) in zip(axes, panels):
                ax.imshow(Image.open(png))
                ax.axis("off")
                ax.set_title(f"step {i}\n{r:.1f} Å from native", color=INK2, fontsize=15)
            fig.tight_layout()
            fig.savefig(out / f"trpcage_{kind}_strip.png", dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (kind, shown, color, title) in zip(axes, [
            ("folding", args.folding_seed, BLUE, "Folding from extended chain, no guidance"),
            ("unfolding", args.unfolding_seed, ORANGE, "Unfolding with guidance")]):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=15)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for seed, r in sorted(traces[kind].items()):
            ax.plot(r, color=color if seed == shown else MUTED, lw=1.6 if seed == shown else 1.0,
                    alpha=1.0 if seed == shown else 0.5, zorder=3 if seed == shown else 2)
        n_end = sum(r[-100:].mean() < 2 for r in traces[kind].values()) if kind == "folding" else \
            sum(r[-100:].mean() > 5 for r in traces[kind].values())
        verb = "folded (< 2 Å)" if kind == "folding" else "unfolded (> 5 Å)"
        ax.set_title(f"{title}\n{n_end}/{len(traces[kind])} seeds {verb}", color=INK, fontsize=16, loc="left")
        ax.set_xlabel("walk-jump step", color=INK2)
    axes[0].set_ylabel("Cα RMSD to native (Å)", color=INK2)
    fig.text(0.5, -0.04, "Trp-cage (1L2Y), 20 residues. Coloured: the run shown in the snapshots; grey: other seeds.",
             ha="center", color=INK2, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "trpcage_rmsd.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
