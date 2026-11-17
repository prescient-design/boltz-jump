#!/usr/bin/env python3
"""Coverage of each walk-jump exploration strategy on the ConforMix dev split.

Reads the runs written by make_diversity_tasks.py
(results/conformix_diversity/<strategy>/<subset>/chain<k>/...) and, per target,
pools the frames of all chains of a strategy (diverse seeding has 5 chains of
200 steps; the others 1 chain of 1000). Frames with a broken backbone -- any
consecutive C-alpha pair further than 4.3 A apart -- are excluded before
scoring, so a strategy cannot gain coverage by tearing the chain apart. The
fraction of valid frames is reported alongside.

Short C-alpha pairs are deliberately not penalised: about 2.9 A is a genuine
cis peptide bond, and 2.0-2.7 A is the bond compression of the one-step jump,
which affects every strategy alike.

Coverage uses the same definition and reference handling as
eval_coverage_kabsch.py: closest-frame Kabsch C-alpha RMSD to each reference
state, divided by the reference-to-reference RMSD.

Usage:
    python conformix_benchmark/scripts/eval_diversity.py --split dev
"""

import argparse
import csv
import os
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from eval_coverage_kabsch import load_dataset, pair_rmsd, reference

HERE = Path(__file__).resolve().parent.parent
CA_BREAK_A = 4.3


def frames_for(root, strategy, subset, target):
    chains = sorted((root / strategy / subset).glob("chain*"))
    xyz = []
    for ch in chains:
        pred = sorted(ch.glob(f"boltz_jump_results_*_{target}/predictions/{target}"))
        if not pred:
            continue
        d = pred[0]
        t = md.load(str(d / f"{target}_model_0.xtc"), top=str(d / f"{target}_model_0.cif"))
        t = t.atom_slice(t.top.select("name CA"))
        xyz.append(10 * t.xyz)
    if not xyz:
        return None, 0
    x = np.concatenate(xyz)
    longest = np.linalg.norm(np.diff(x, axis=1), axis=-1).max(1)
    return x[longest < CA_BREAK_A], len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ref_dir", default=str(HERE.parent / "conformix_ref_pdbs"))
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True), override=False)
    root = Path(os.environ["RESULTS_DIR"]) / "conformix_diversity"
    split = {(r["subset"], r["target"]): r["split"] for r in csv.DictReader(open(HERE / "splits.csv"))}
    strategies = sorted(p.name for p in root.iterdir() if p.is_dir())

    rows = []
    for subset in ["crypticpockets", "domainmotion", "foldswitching", "membranetransporters"]:
        systems = load_dataset(subset)
        for s in systems.itertuples():
            if split.get((subset, s.system_id)) != args.split:
                continue
            try:
                r1, r2 = reference(s.pdb1, s.sequence, args.ref_dir), reference(s.pdb2, s.sequence, args.ref_dir)
            except Exception as e:
                print(f"  skipping {s.system_id}: {e}")
                continue
            for strategy in strategies:
                x, n_all = frames_for(root, strategy, subset, s.system_id)
                if x is None:
                    continue
                row = {"subset": subset, "target": s.system_id, "strategy": strategy,
                       "frames": n_all, "valid_frac": len(x) / n_all}
                if len(x) and x.shape[1] == len(s.sequence):
                    d1, d2 = pair_rmsd(x, r1).min(), pair_rmsd(x, r2).min()
                    row.update(best_frac=min(d1, d2) / s.RMSD, alt_frac=max(d1, d2) / s.RMSD)
                rows.append(row)

    df = pd.DataFrame(rows)
    out = HERE / "eval_kabsch" / f"diversity_{args.split}.csv"
    df.to_csv(out, index=False)
    scored = df.dropna(subset=["best_frac"])
    run = sorted(scored.strategy.unique())  # strategies actually run on this split
    # A protein can appear in two subsets with different reference states, so a
    # benchmark system is (subset, target), not the target name alone.
    common = scored.groupby(["subset", "target"]).strategy.nunique()
    keep = set(common[common == len(run)].index)  # systems every strategy covers
    d = scored[[k in keep for k in zip(scored.subset, scored.target)]]
    print(f"{len(keep)} {args.split} systems covered by all {len(run)} strategies run on this split: {run}\n")
    print(f"{'strategy':<20}{'best@50%':>10}{'alt@50%':>10}{'valid frames':>14}")
    for strategy, g in d.groupby("strategy"):
        print(f"{strategy:<20}{(g.best_frac <= 0.5).mean():>10.2f}{(g.alt_frac <= 0.5).mean():>10.2f}"
              f"{g.valid_frac.mean():>14.2f}")
    print(f"\nper-target results -> {out}")


if __name__ == "__main__":
    main()
