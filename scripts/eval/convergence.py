#!/usr/bin/env python3
"""Divergence from the MD reference as a function of the number of samples.

For one target, every model's ensemble is truncated to its first N samples
(N in --sizes) and compared with the MD reference by RMWD, the root mean
Wasserstein distance used in the paper: per heavy atom, the Gaussian
2-Wasserstein distance between predicted and reference positions,

    W_i^2 = |mu_p - mu_r|^2 + tr(S_p + S_r - 2 (S_r^1/2 S_p S_r^1/2)^1/2),

aggregated as sqrt(rms_i(translation)^2 + rms_i(spread)^2), the same as
analyse_ensembles.py + plot.py. Preprocessing matches analyse_ensembles.py:
hydrogens removed, atoms matched by name, everything superposed on the crystal
structure, and the reference is 1000 random MD frames (seed 137).

"First N" means the first N frames of a walk-jump chain (consecutive and
correlated) and the first N of the independent samples for the other models,
which is what each method costs per sample.

--seeded_atoms_only drops atoms whose CHARMM name in the reference PDB (Ile CD,
C-terminal OT1/OT2) is renamed by mdtraj: Boltz's PDB parser doesn't recognise
these names, so they are missing from the walk-jump seed and take ~30 steps to
be pulled into place, which dominates the RMWD of short walk-jump chains.

Usage:
    python scripts/eval/convergence.py --dataset atlas_test --name 1d3y_B \
        --reference_dir $ATLAS_DIR --predictions_dir results_organized/atlas_test_msa --out conv.csv
"""

import argparse
from pathlib import Path

import mdtraj
import numpy as np
import pandas as pd

from analyse_ensembles import align_tops

MODELS = ["boltz2_jump_msa_d1", "boltz2_jump_msa_no_init_d1", "boltz2_full_msa", "boltz2_msa_subsample_32", "alphaflow_md_base", "bioemu"]


def load_reference(dataset, name, ref_dir):
    if dataset == "atlas_test":
        top = f"{ref_dir}/{name}/protein/{name}.pdb"
        traj = mdtraj.join([mdtraj.load(f"{ref_dir}/{name}/protein/{name}_prod_R{r}_fit.xtc", top=top) for r in (1, 2, 3)])
    else:
        top = f"{ref_dir}/{name}.pdb"
        traj = mdtraj.join([mdtraj.load(f"{ref_dir}/{name}_320_{r}.xtc", top=top) for r in range(5)])
    return top, mdtraj.load(top), traj


def seeded_atoms(pdb, top):
    """Atoms of top (loaded from pdb, same order) whose PDB name mdtraj kept unchanged."""
    raw = [line[12:16].strip() for line in open(pdb) if line.startswith(("ATOM", "HETATM"))]
    assert len(raw) == top.n_atoms, (len(raw), top.n_atoms)
    return {a.index for a in top.atoms if a.name == raw[a.index]}


def renumber_and_strip(t, keep=None):
    for i, residue in enumerate(t.top.residues, start=1):
        residue.resSeq = i
    return t.atom_slice([a.index for a in t.top.atoms
                         if a.element.symbol != "H" and (keep is None or a.index in keep)])


def mean_cov(xyz):
    mu = xyz.mean(0)
    d = xyz - mu
    return mu, np.einsum("fai,faj->aij", d, d) / len(xyz)


def psd_sqrt(m):
    w, v = np.linalg.eigh(m)
    return (v * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.swapaxes(v, -1, -2)


def rmwd(ref_mu, ref_cov, ref_sqrt, xyz):
    mu, cov = mean_cov(xyz)
    trans = np.linalg.norm(mu - ref_mu, axis=-1)
    cross = psd_sqrt(ref_sqrt @ cov @ ref_sqrt)
    spread2 = np.trace(ref_cov + cov - 2 * cross, axis1=1, axis2=2)
    spread = np.sqrt(np.clip(spread2, 0, None))
    # nm -> A; returns (RMWD, rms translation term, rms spread term)
    t, sp = 10 * np.sqrt(np.mean(trans**2)), 10 * np.sqrt(np.mean(spread**2))
    return float(np.hypot(t, sp)), float(t), float(sp)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["atlas_test", "mdcath"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--seeded_atoms_only", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    top, crystal, md = load_reference(args.dataset, args.name, args.reference_dir)
    keep = seeded_atoms(top, crystal.top) if args.seeded_atoms_only else None
    crystal, md = renumber_and_strip(crystal, keep), renumber_and_strip(md, keep)
    rng = np.random.RandomState(137)
    md_sample = md[rng.randint(0, md.n_frames, 1000)]

    rows = []
    for model in MODELS:
        d = Path(args.predictions_dir) / args.name / model
        if not (d / "traj.xtc").exists():
            continue
        pred = renumber_and_strip(mdtraj.load(str(d / "traj.xtc"), top=str(d / "system.cif")))
        refmask, predmask = align_tops(md_sample.top, pred.top)
        if len(predmask) <= 0.8 * len(refmask) or len(refmask) <= 0.8 * md_sample.n_atoms:
            continue
        ref_c, ref_t, pred_t = crystal.atom_slice(refmask), md_sample.atom_slice(refmask), pred.atom_slice(predmask)
        ref_t.superpose(ref_c)
        pred_t.superpose(ref_c)
        ref_mu, ref_cov = mean_cov(ref_t.xyz)
        ref_sqrt = psd_sqrt(ref_cov)
        for n in args.sizes:
            if n > pred_t.n_frames:
                continue
            rows.append({"dataset": args.dataset, "target": args.name, "model": model, "n": n,
                         "atoms": "seeded" if args.seeded_atoms_only else "all",
                         **dict(zip(["rmwd", "translation", "spread"], rmwd(ref_mu, ref_cov, ref_sqrt, pred_t.xyz[:n]))),
                         "ref_spread": 10 * float(np.sqrt(np.mean(np.trace(ref_cov, axis1=1, axis2=2)))),
                         "pred_spread": 10 * float(np.sqrt(np.mean(np.trace(mean_cov(pred_t.xyz[:n])[1], axis1=1, axis2=2))))})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{args.name}: {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
