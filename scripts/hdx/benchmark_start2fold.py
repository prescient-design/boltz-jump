#!/usr/bin/env python3
"""Native walk-jump vs static structure on every prepared Start2Fold entry.

Subcommands:

    tasks   write the SLURM task list: native walk-jump (sigma = 2 A, delta = 1,
            friction 0.1, 1000 steps) from the experimental structure, 3 seeds per
            entry, for every entry marked ready by prepare_start2fold.py.
    score   score every entry and aggregate.

Per entry, the static structure and the walk-jump ensemble (ln P averaged over
the 3 seeds) are scored with protection_factors.py. Only residues the
experiment classified are compared -- unlabelled residues mix fast-exchanging
and unmeasured amides, so they are left out. Three pairwise tests are made
wherever both classes are present: STRONG > MEDIUM, STRONG > WEAK and
MEDIUM > WEAK. For each, the AUC gain of the ensemble over the static structure
is bootstrapped over residues (paired). Entries reporting a single class have
nothing to compare and are recorded as such.

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
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from hdx_labels import experiment_sets, labels_for_model

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
COMPARISONS = (("STRONG", "MEDIUM"), ("STRONG", "WEAK"), ("MEDIUM", "WEAK"))
WJ_ARGS = ("--walk_jump_sigma 2.0 --walk_jump_delta 1.0 --walk_jump_friction 0.1 "
           "--walk_jump_steps 1000 --no_kernels --override")
# Sampling strategies: the native walk, and the same walk with an RMSD repulsion
# from the starting structure (seed repulsion; see wrappers/diversity.py).
STRATEGIES = {"native": ""}
STRATEGIES.update({f"repel_h{h}": f"--diversity_mode seed --diversity_height {h} --diversity_width 2"
                   for h in ("0.04", "0.08", "0.15")})


def runs_dir(root, strategy):
    return root / "runs" if strategy == "native" else root / f"runs_{strategy}"


def scorable(r):
    """Entries reporting at least two protection classes can be compared (refined per experiment later)."""
    return "|" in r["classes"]


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


REUSE = False  # set by --reuse: read existing per-residue CSVs instead of rescoring


def run_scorer(entry_xml, out_csv, topology, trajectory=None):
    if REUSE and Path(out_csv).exists():
        return pd.read_csv(out_csv).set_index("residue")
    cmd = [sys.executable, str(HERE / "protection_factors.py"), "--entry", str(entry_xml),
           "--topology", str(topology), "--out", str(out_csv), "--stride", "5"]
    if trajectory:
        cmd += ["--trajectory", str(trajectory)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return pd.read_csv(out_csv).set_index("residue")


def to_model_numbering(series, entry_xml, model_sequence):
    """protection_factors.py numbers residues on the entry's first sequence; renumber 1..len(model)."""
    first = next(ET.parse(entry_xml).getroot().iter("sequence")).text.strip()
    offset = first.find(model_sequence)
    out = series.copy()
    out.index = out.index - offset
    return out


def model_sequence(root, eid):
    return re.search(r"sequence: (\S+)", (root / "configs" / f"{eid}.yaml").read_text()).group(1)


def ensemble(root, scores, eid, entry_xml, strategy):
    """ln P per residue averaged over the seeds of one strategy, or None."""
    seeds = []
    for s in SEEDS:
        pred = sorted((runs_dir(root, strategy) / eid / f"seed{s}").glob("boltz_jump_results_*/predictions/*"))
        if not pred:
            continue
        d = pred[0]
        tag = "" if strategy == "native" else f"_{strategy}"
        seeds.append(run_scorer(entry_xml, scores / f"{eid}{tag}_seed{s}.csv", d / f"{d.name}_model_0.cif",
                                d / f"{d.name}_model_0.xtc").lnP)
    return (pd.concat(seeds, axis=1).mean(axis=1), len(seeds)) if seeds else (None, 0)


def score_entry(r, root, scores, rng, strategy="native", versus="static"):
    eid = r["entry"]
    entry_xml = root / "downloads" / f"{eid}.xml"
    uniprot = ET.parse(entry_xml).getroot().find("protein").get("uniprot_id")
    static = run_scorer(entry_xml, scores / f"{eid}_static.csv", root / "structures" / f"{eid}.cif")
    ens, n_seeds = ensemble(root, scores, eid, entry_xml, strategy)
    if ens is None:
        return {"entry": eid, "uniprot": uniprot, "name": r["name"], "note": f"no {strategy} runs"}
    reference = static.lnP
    if versus == "native":
        reference, _ = ensemble(root, scores, eid, entry_xml, "native")
        if reference is None:
            return {"entry": eid, "uniprot": uniprot, "name": r["name"], "note": "no native runs"}
    seq = model_sequence(root, eid)
    ens, reference = to_model_numbering(ens, entry_xml, seq), to_model_numbering(reference, entry_xml, seq)
    sets = labels_for_model(experiment_sets(entry_xml, "stability"), seq)
    row = {"entry": eid, "uniprot": uniprot, "name": r["name"], "n_seeds": n_seeds, "n_sets": len(sets)}
    # Each experiment is compared on its own tiers; an entry's gain is the mean over its experiments.
    for hi_c, lo_c in COMPARISONS:
        key = f"{hi_c.lower()}_vs_{lo_c.lower()}"
        vals = []
        for _, levels in sets:
            lv = pd.Series(levels).reindex(ens.index)
            pos, neg = lv.index[lv == hi_c], lv.index[lv == lo_c]
            if len(pos) and len(neg):
                g, lo, hi = paired_gain(ens, reference, pos, neg, rng)
                vals.append((auc(reference[pos], reference[neg]), auc(ens[pos], ens[neg]), g, lo, hi))
        if vals:
            v = np.array(vals).mean(0)
            row.update({f"{key}_static": v[0], f"{key}_wj": v[1], f"{key}_gain": v[2], f"{key}_lo": v[3], f"{key}_hi": v[4]})
    if not any(k.endswith("_gain") for k in row):
        row["note"] = "only one protection class"
    gains = "  ".join(f"{h[0]}>{l[0]} {row[f'{h.lower()}_vs_{l.lower()}_gain']:+.2f}"
                      for h, l in COMPARISONS if f"{h.lower()}_vs_{l.lower()}_gain" in row)
    print(f"{eid} {r['name'][:30]:<30} {gains or row.get('note', '')}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tasks", "score"])
    parser.add_argument("--hdx_dir", default=os.environ.get("HDX_DIR"))
    parser.add_argument("--reuse", action="store_true", help="reuse existing per-residue scores")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="native")
    parser.add_argument("--versus", choices=["static", "native"], default="static",
                        help="what the strategy is compared against")
    parser.add_argument("--scorable_only", action="store_true",
                        help="tasks: only entries with at least two protection classes")
    args = parser.parse_args()
    global REUSE
    REUSE = args.reuse
    root = Path(args.hdx_dir) / "start2fold"
    manifest = [r for r in csv.DictReader(open(root / "manifest.csv")) if r["status"] == "ready"]

    if args.command == "tasks":
        entries = [r for r in manifest if scorable(r)] if args.scorable_only else manifest
        extra = STRATEGIES[args.strategy]
        lines = [f"{root / 'configs' / (r['entry'] + '.yaml')}\t{runs_dir(root, args.strategy) / r['entry'] / f'seed{s}'}"
                 f"\t{s}\t{WJ_ARGS} {extra}".rstrip() + "\n" for r in entries for s in SEEDS]
        out = root / ("tasks.tsv" if args.strategy == "native" else f"tasks_{args.strategy}.tsv")
        out.write_text("".join(lines))
        print(f"{len(entries)} entries x {len(SEEDS)} seeds = {len(lines)} runs -> {out}")
        return

    rng = np.random.default_rng(0)
    scores = root / "scores"
    scores.mkdir(exist_ok=True)
    rows = []
    print(f"strategy {args.strategy} vs {args.versus}")
    for r in [r for r in manifest if scorable(r)]:
        try:
            rows.append(score_entry(r, root, scores, rng, args.strategy, args.versus))
        except Exception as e:  # one entry must not stop the benchmark; the reason is kept
            detail = getattr(e, "stderr", "") or str(e)
            rows.append({"entry": r["entry"], "name": r["name"], "note": detail.strip().splitlines()[-1][:200]})
            print(f"{r['entry']} FAILED: {rows[-1]['note']}")
    df = pd.DataFrame(rows)
    out_csv = root / f"benchmark_{args.strategy}_vs_{args.versus}.csv"
    df.to_csv(out_csv, index=False)
    def summarise(label, per_protein):
        wins, n = int((per_protein > 0).sum()), int((per_protein != 0).sum())
        p = binomtest(wins, n, 0.5, alternative="greater").pvalue if n else float("nan")
        print(f"{label:<16} {len(per_protein):>3} proteins  mean gain {per_protein.mean():+.3f}  "
              f"median {per_protein.median():+.3f}  walk-jump better on {wins}/{n} (sign test p = {p:.3g})")

    print()
    gain_cols = []
    for hi_c, lo_c in COMPARISONS:
        col = f"{hi_c.lower()}_vs_{lo_c.lower()}_gain"
        if col in df and df[col].notna().any():
            gain_cols.append(col)
            summarise(f"{hi_c} > {lo_c}", df.dropna(subset=[col]).groupby("uniprot")[col].mean())
    # all comparisons together: each protein's mean gain over the pairs it has
    # (dropna: recent pandas keeps missing values when stacking)
    pooled = df.set_index("uniprot")[gain_cols].stack().dropna().groupby(level=0).mean()
    summarise("all pairs", pooled)
    print(f"\nper-entry results -> {out_csv}")


if __name__ == "__main__":
    main()
