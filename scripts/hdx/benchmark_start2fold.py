#!/usr/bin/env python3
"""Native walk-jump vs static structure on every prepared Start2Fold entry.

Subcommands:

    tasks   write the SLURM task list: native walk-jump (sigma = 2 A, delta = 1,
            friction 0.1, 1000 steps) from the experimental structure, 3 seeds per
            entry, for every entry marked ready by prepare_start2fold.py.
    score   score every entry and aggregate.

Per entry, the static structure and the walk-jump ensemble (ln P averaged over
the 3 seeds) are scored with protection_factors.py, and the AUC gain of the
ensemble over the static structure is bootstrapped over residues (paired). The
primary metric is STRONG > MEDIUM where an entry has both classes; entries with
a single class contribute labelled > unlabelled only, reported separately.

Several proteins appear in more than one entry (e.g. lysozymes from different
species are distinct, but repeated UniProt IDs are not). Entries are grouped by
UniProt ID and each protein's gains are averaged, so a protein with two entries
counts once. The summary is a sign test over proteins plus the mean gain.

Usage:
    HDX_DIR=/path python scripts/hdx/benchmark_start2fold.py tasks
    HDX_DIR=/path python scripts/hdx/benchmark_start2fold.py score
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

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
WJ_ARGS = ("--walk_jump_sigma 2.0 --walk_jump_delta 1.0 --walk_jump_friction 0.1 "
           "--walk_jump_steps 1000 --no_kernels --override")


def auc(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float((x[:, None] > y[None]).mean() + 0.5 * (x[:, None] == y[None]).mean())


def paired_gain(ens, static, pos, neg, rng, n_boot=4000):
    gain = auc(ens[pos], ens[neg]) - auc(static[pos], static[neg])
    boots = []
    for _ in range(n_boot):
        p, n = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        boots.append(auc(ens[p], ens[n]) - auc(static[p], static[n]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return gain, lo, hi


def run_scorer(entry_xml, out_csv, topology, trajectory=None):
    cmd = [sys.executable, str(HERE / "protection_factors.py"), "--entry", str(entry_xml),
           "--topology", str(topology), "--out", str(out_csv), "--stride", "5"]
    if trajectory:
        cmd += ["--trajectory", str(trajectory)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return pd.read_csv(out_csv).set_index("residue")


def score_entry(r, root, scores, rng):
    eid = r["entry"]
    entry_xml = root / "downloads" / f"{eid}.xml"
    uniprot = ET.parse(entry_xml).getroot().find("protein").get("uniprot_id")
    static = run_scorer(entry_xml, scores / f"{eid}_static.csv", root / "structures" / f"{eid}.cif")
    seeds = []
    for s in SEEDS:
        pred = sorted((root / "runs" / eid / f"seed{s}").glob("boltz_jump_results_*/predictions/*"))
        if not pred:
            continue
        d = pred[0]
        seeds.append(run_scorer(entry_xml, scores / f"{eid}_seed{s}.csv", d / f"{d.name}_model_0.cif",
                                d / f"{d.name}_model_0.xtc").lnP)
    if not seeds:
        return {"entry": eid, "uniprot": uniprot, "name": r["name"], "note": "no walk-jump runs"}
    ens = pd.concat(seeds, axis=1).mean(axis=1)
    lv = static.hdx_level
    strong, medium = lv.index[lv == "STRONG"], lv.index[lv == "MEDIUM"]
    protected, weak = lv.index[lv.isin(["STRONG", "MEDIUM"])], lv.index[lv == "WEAK"]
    negative = weak if len(weak) else lv.index[lv == "unlabelled"]
    row = {"entry": eid, "uniprot": uniprot, "name": r["name"], "n_seeds": len(seeds),
           "n_strong": len(strong), "n_medium": len(medium), "n_weak": len(weak),
           "secondary_versus": "WEAK" if len(weak) else "unlabelled"}
    if len(strong) and len(medium):
        g, lo, hi = paired_gain(ens, static.lnP, strong, medium, rng)
        row.update(primary_static=auc(static.lnP[strong].values, static.lnP[medium].values),
                   primary_wj=auc(ens[strong].values, ens[medium].values), primary_gain=g,
                   primary_lo=lo, primary_hi=hi)
    if len(protected) and len(negative):
        g, lo, hi = paired_gain(ens, static.lnP, protected, negative, rng)
        row.update(secondary_static=auc(static.lnP[protected], static.lnP[negative]),
                   secondary_wj=auc(ens[protected], ens[negative]), secondary_gain=g,
                   secondary_lo=lo, secondary_hi=hi)
    print(f"{eid} {r['name'][:30]:<30} primary {row.get('primary_gain', float('nan')):+.2f}  "
          f"secondary {row.get('secondary_gain', float('nan')):+.2f}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tasks", "score"])
    parser.add_argument("--hdx_dir", default=os.environ.get("HDX_DIR"))
    args = parser.parse_args()
    root = Path(args.hdx_dir) / "start2fold"
    manifest = [r for r in csv.DictReader(open(root / "manifest.csv")) if r["status"] == "ready"]

    if args.command == "tasks":
        lines = [f"{root / 'configs' / (r['entry'] + '.yaml')}\t{root / 'runs' / r['entry'] / f'seed{s}'}\t{s}\t{WJ_ARGS}\n"
                 for r in manifest for s in SEEDS]
        (root / "tasks.tsv").write_text("".join(lines))
        print(f"{len(manifest)} entries x {len(SEEDS)} seeds = {len(lines)} runs -> {root / 'tasks.tsv'}")
        return

    rng = np.random.default_rng(0)
    scores = root / "scores"
    scores.mkdir(exist_ok=True)
    rows = []
    for r in manifest:
        try:
            rows.append(score_entry(r, root, scores, rng))
        except Exception as e:  # one entry must not stop the benchmark; the reason is kept
            detail = getattr(e, "stderr", "") or str(e)
            rows.append({"entry": r["entry"], "name": r["name"], "note": detail.strip().splitlines()[-1][:200]})
            print(f"{r['entry']} FAILED: {rows[-1]['note']}")
    df = pd.DataFrame(rows)
    df.to_csv(root / "benchmark_entries.csv", index=False)
    for metric in ("primary", "secondary"):
        col = f"{metric}_gain"
        if col not in df:
            continue
        per_protein = df.dropna(subset=[col]).groupby("uniprot")[col].mean()
        if per_protein.empty:
            continue
        wins, n = int((per_protein > 0).sum()), int((per_protein != 0).sum())
        p = binomtest(wins, n, 0.5, alternative="greater").pvalue if n else float("nan")
        print(f"\n{metric}: {len(per_protein)} proteins, mean gain {per_protein.mean():+.3f}, "
              f"median {per_protein.median():+.3f}; walk-jump better on {wins}/{n} (sign test p = {p:.3g})")
    print(f"\nper-entry results -> {root / 'benchmark_entries.csv'}")


if __name__ == "__main__":
    main()
