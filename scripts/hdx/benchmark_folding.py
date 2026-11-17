#!/usr/bin/env python3
"""Does protection acquired while collapsing from random coil predict folding HDX?

For each entry with at least two folding classes (EARLY > INTERMEDIATE > LATE,
from pulse-labelling or quenched-flow HDX), predicted protection is averaged
over a window of the walk-jump trajectories started from random coil
(prepare_folding.py), pooled over starts. Pre-specified: the first half of each
trajectory (the collapse) is primary, the whole trajectory secondary.

Two questions, both asked with the pairwise AUCs EARLY > INTERMEDIATE,
EARLY > LATE and INTERMEDIATE > LATE over classified residues only:

1. Does the collapse trajectory rank folding classes at all? (AUC vs 0.5)
2. Does it beat the native static structure? Early-folding residues often form
   the stable core, so the native structure is the control that matters.
   (paired bootstrap of the AUC difference)

Each protein (UniProt ID) counts once; summaries are sign tests over proteins.

Usage:
    HDX_DIR=/path python scripts/hdx/benchmark_folding.py --window 0:0.5
"""

import argparse
import csv
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from benchmark_start2fold import auc, model_sequence, paired_gain, to_model_numbering
from hdx_labels import experiment_sets, labels_for_model

HERE = Path(__file__).resolve().parent
PAIRS = (("EARLY", "INTERMEDIATE"), ("EARLY", "LATE"), ("INTERMEDIATE", "LATE"))


def score(entry_xml, out_csv, topology, trajectory=None, window="0:1", reuse=False):
    if not (reuse and out_csv.exists()):
        cmd = [sys.executable, str(HERE / "protection_factors.py"), "--entry", str(entry_xml), "--topology",
               str(topology), "--out", str(out_csv), "--classes", "folding", "--window", window, "--stride", "5"]
        if trajectory:
            cmd += ["--trajectory", str(trajectory)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    return pd.read_csv(out_csv).set_index("residue")


def sign_test(values, label):
    v = pd.Series(values).dropna()
    wins, n = int((v > 0).sum()), int((v != 0).sum())
    p = binomtest(wins, n, 0.5, alternative="greater").pvalue if n else float("nan")
    print(f"  {label:<44} {len(v):>3} proteins  mean {v.mean():+.3f}  better on {wins}/{n}  (p = {p:.3g})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdx_dir", default=os.environ.get("HDX_DIR"))
    parser.add_argument("--window", default="0:0.5")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--runs", default="runs", help="runs (unguided) or runs_guided")
    args = parser.parse_args()
    root = Path(args.hdx_dir) / "start2fold"
    fold = root / "folding"
    scores = fold / "scores"
    scores.mkdir(exist_ok=True)
    tag = f"{args.runs}_{args.window.replace(':', '-')}"
    rng = np.random.default_rng(0)

    runs_dir = fold / args.runs
    entries = sorted({p.parent.name for p in runs_dir.glob("*/coil*")})
    rows = []
    for eid in entries:
        entry_xml = root / "downloads" / f"{eid}.xml"
        uniprot = ET.parse(entry_xml).getroot().find("protein").get("uniprot_id")
        try:
            static = score(entry_xml, scores / f"{eid}_static.csv", root / "structures" / f"{eid}.cif", reuse=args.reuse)
            runs = []
            for coil in sorted((runs_dir / eid).glob("coil*")):
                pred = sorted(coil.glob("boltz_jump_results_*/predictions/*"))
                if not pred:
                    continue
                d = pred[0]
                runs.append(score(entry_xml, scores / f"{eid}_{coil.name}_{tag}.csv", d / f"{d.name}_model_0.cif",
                                  d / f"{d.name}_model_0.xtc", args.window, args.reuse).lnP)
        except subprocess.CalledProcessError as e:
            print(f"{eid} FAILED: {e.stderr.strip().splitlines()[-1][:150]}")
            continue
        if not runs:
            continue
        traj = pd.concat(runs, axis=1).mean(axis=1)
        seq = model_sequence(root, eid)
        traj, nat = to_model_numbering(traj, entry_xml, seq), to_model_numbering(static.lnP, entry_xml, seq)
        sets = labels_for_model(experiment_sets(entry_xml, "folding"), seq)
        row = {"entry": eid, "uniprot": uniprot, "n_starts": len(runs), "n_sets": len(sets)}
        for hi_c, lo_c in PAIRS:
            key = f"{hi_c[0]}>{lo_c[0]}"
            vals = []
            for _, levels in sets:  # each experiment on its own tiers; mean over experiments
                lv = pd.Series(levels).reindex(traj.index)
                pos, neg = lv.index[lv == hi_c], lv.index[lv == lo_c]
                if len(pos) and len(neg):
                    g, _, _ = paired_gain(traj, nat, pos, neg, rng)
                    vals.append((auc(traj[pos], traj[neg]), auc(nat[pos], nat[neg]), g))
            if vals:
                v = np.array(vals).mean(0)
                row.update({f"{key}_traj": v[0], f"{key}_static": v[1], f"{key}_gain": v[2]})
        rows.append(row)
        print(f"{eid} " + "  ".join(f"{k}: traj {row[f'{k}_traj']:.2f} static {row[f'{k}_static']:.2f}"
                                    for k in ("E>I", "E>L", "I>L") if f"{k}_traj" in row))

    df = pd.DataFrame(rows)
    df.to_csv(fold / f"benchmark_folding_{tag}.csv", index=False)
    print(f"\n{args.runs}, window {args.window}; {len(df)} entries")
    keys = [k for k in ("E>I", "E>L", "I>L") if f"{k}_traj" in df]
    per = lambda col: df.dropna(subset=[col]).groupby("uniprot")[col].mean()
    pooled = lambda suffix: (df.set_index("uniprot")[[f"{k}{suffix}" for k in keys]].stack().dropna()
                             .groupby(level=0).mean())
    print("1. collapse trajectory vs chance (AUC - 0.5):")
    for k in keys:
        sign_test(per(f"{k}_traj") - 0.5, k)
    sign_test(pooled("_traj") - 0.5, "all pairs")
    print("   native static structure vs chance, for reference:")
    sign_test(pooled("_static") - 0.5, "all pairs")
    print("2. collapse trajectory vs native static structure (AUC gain):")
    for k in keys:
        sign_test(per(f"{k}_gain"), k)
    sign_test(pooled("_gain"), "all pairs")


if __name__ == "__main__":
    main()
