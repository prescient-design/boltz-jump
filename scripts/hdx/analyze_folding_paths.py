#!/usr/bin/env python3
"""Compare folding pathways across guidance schemes.

For each condition (a directory of runs from different random-coil starts):

- fold:        fraction of starts reaching Q >= 0.8, and the median first step to do so
- agreement:   how consistently residues fold in the same order from different
               starts -- mean Spearman correlation of per-residue onset between
               pairs of starts. A reproducible pathway has high agreement.
- restraint:   correlation of onset with each residue's starting distance from
               its native position. A bias that pulls residues individually makes
               far residues fold late; an order-agnostic bias should not.

Q is the fraction of native C-alpha contacts (within 8 A in the target,
|i - j| >= 4), a contact counting as formed below 1.5x its native distance. A
residue's onset is the first step from which at least half its native contacts
stay formed for 10 consecutive saved frames.

Usage:
    python scripts/hdx/analyze_folding_paths.py --target 1pga.cif --runs runs_fold_1pga/guided_w0.003 \
        runs_fold_1pga/qratchet_k5 ...
"""

import argparse
import glob
from itertools import combinations
from pathlib import Path

import mdtraj as md
import numpy as np
from scipy.stats import spearmanr


def native_contacts(ref):
    ca = ref.topology.select("name CA")
    x = ref.xyz[0, ca]
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    i, j = np.triu_indices(len(ca), k=4)
    m = d[i, j] < 0.8
    return ca, i[m], j[m], d[i, j][m]


def pathway(xtc, ref, rca, ii, jj, d0, stride):
    t = md.load(xtc, top=xtc.replace(".xtc", ".cif"), stride=stride)
    ca = t.topology.select("name CA")
    formed = np.linalg.norm(t.xyz[:, ca][:, ii] - t.xyz[:, ca][:, jj], axis=-1) < 1.5 * d0
    q = formed.mean(1)
    onset = np.full(len(rca), np.nan)
    for r in range(len(rca)):
        c = (ii == r) | (jj == r)
        if c.sum() < 2:
            continue
        held = np.convolve(formed[:, c].mean(1) >= 0.5, np.ones(10, int), "valid") >= 10
        if held.any():
            onset[r] = stride * np.argmax(held)
    t0 = t[0]
    t0.superpose(ref, atom_indices=ca, ref_atom_indices=rca)
    start_dist = np.linalg.norm(t0.xyz[0, ca] - ref.xyz[0, rca], axis=-1)
    folded = stride * int(np.argmax(q >= 0.8)) if (q >= 0.8).any() else None
    return onset, start_dist, folded, q[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    ref = md.load(args.target)
    ref = ref.atom_slice(ref.topology.select("protein"))
    rca, ii, jj, d0 = native_contacts(ref)

    print(f"{'condition':<34}{'folded':>8}{'median step':>13}{'final Q':>9}{'agreement':>11}{'restraint':>11}")
    for run in args.runs:
        files = sorted(glob.glob(f"{run}/start*/boltz_jump_results_*/predictions/*/*_model_0.xtc"))
        if not files:
            print(f"{Path(run).name:<34} no runs")
            continue
        res = [pathway(f, ref, rca, ii, jj, d0, args.stride) for f in files]
        steps = [r[2] for r in res if r[2] is not None]
        agree = [spearmanr(a[0], b[0], nan_policy="omit")[0] for a, b in combinations(res, 2)]
        restr = [spearmanr(o, d, nan_policy="omit")[0] for o, d, *_ in res]
        print(f"{Path(run).name:<34}{len(steps):>5}/{len(res):<2}"
              f"{(np.median(steps) if steps else float('nan')):>13.0f}{np.median([r[3] for r in res]):>9.2f}"
              f"{np.nanmean(agree):>11.2f}{np.nanmean(restr):>11.2f}")


if __name__ == "__main__":
    main()
