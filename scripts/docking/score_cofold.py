#!/usr/bin/env python3
"""Score PoseBusters co-folding samples and select pose-refinement targets.

For every co-folded sample (scripts/slurm/docking_cofold.sh), computes the
pocket-aligned symmetry-aware ligand RMSD, pocket C-alpha RMSD and pocket
Jaccard against the crystal structure, plus Boltz-2 confidences. An entry is a
pose-refinement target ("clean pose failure") if, at its best-ligand sample,

    ligand RMSD > 2 A, pocket C-alpha RMSD <= 2 A and pocket Jaccard >= 0.3,

i.e. the protein is folded correctly and the ligand is in the right pocket but
mis-docked.

Writes $DOCKING_DIR/posebusters/cofold{RUN_TAG}_scores.csv and, for the
5-sample screen, failures.csv and failure_configs.txt (inputs to the
matched-budget baseline).

Usage:
    python scripts/docking/score_cofold.py                    # screen
    python scripts/docking/score_cofold.py --run_tag _600     # matched-budget baseline
"""

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import mdtraj as md

from common import SUCCESS_RMSD_A, PoseScorer, Reference, docking_dir

POCKET_CA_MAX_A = 2.0
JACCARD_MIN = 0.3


def score_entry(args):
    entry, cofold_dir, bench = args
    pred_dirs = sorted(cofold_dir.glob(f"boltz_results_*_{entry}/predictions/{entry}"))
    if not pred_dirs:
        return []
    pred_dir = pred_dirs[0]
    models = sorted(pred_dir.glob(f"{entry}_model_*.cif"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    ref = Reference.posebusters(bench / entry)
    rows = []
    for cif in models:
        k = int(cif.stem.rsplit("_", 1)[1])
        try:
            scorer = PoseScorer(ref, cif)
            s = scorer.score(md.load(str(cif)).xyz[0] * 10.0)
        except ValueError as e:
            print(f"  {entry} model {k}: {e}")
            continue
        conf_path = pred_dir / f"confidence_{entry}_model_{k}.json"
        conf = json.load(open(conf_path)) if conf_path.exists() else {}
        rows.append({"entry": entry, "model": k, "lig_rmsd": round(s["lig_rmsd"], 3),
                     "pocket_ca": round(s["pocket_ca"], 3), "jaccard": round(s["jaccard"], 3),
                     "confidence_score": conf.get("confidence_score"), "iptm": conf.get("iptm"),
                     "ligand_iptm": conf.get("ligand_iptm"), "cif": str(cif)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = docking_dir()
    pb = root / "posebusters"
    bench = root / "data/posebusters/posebusters_benchmark_set"
    manifest = json.load(open(pb / "manifest.json"))["entries"]
    cofold_dir = pb / f"cofold{args.run_tag}"
    with Pool(args.workers) as pool:
        results = pool.map(score_entry, [(e, cofold_dir, bench) for e in sorted(manifest)])
    rows = [r for rs in results for r in rs]
    fields = ["entry", "model", "lig_rmsd", "pocket_ca", "jaccard", "confidence_score", "iptm", "ligand_iptm", "cif"]
    with open(pb / f"cofold{args.run_tag}_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_entries = len({r["entry"] for r in rows})
    print(f"scored {len(rows)} samples of {n_entries}/{len(manifest)} entries")
    if args.run_tag:
        return

    by_entry = {}
    for r in rows:
        by_entry.setdefault(r["entry"], []).append(r)
    verdicts = {"ok": [], "clean_pose_failure": [], "wrong_pocket": [], "fold_failure": []}
    out = []
    for entry, samples in sorted(by_entry.items()):
        best = min(samples, key=lambda r: r["lig_rmsd"])
        if best["lig_rmsd"] <= SUCCESS_RMSD_A:
            verdict = "ok"
        elif best["pocket_ca"] > POCKET_CA_MAX_A:
            verdict = "fold_failure"
        elif best["jaccard"] < JACCARD_MIN:
            verdict = "wrong_pocket"
        else:
            verdict = "clean_pose_failure"
        verdicts[verdict].append(entry)
        # top-ranked (label-free) sample, for reference; walk-jump starts from every sample (prepare_walkjump.py)
        top = max(samples, key=lambda r: (r["confidence_score"] or 0.0))
        out.append({"entry": entry, "verdict": verdict, "n_samples": len(samples),
                    "best_lig_rmsd": best["lig_rmsd"], "pocket_ca_at_best": best["pocket_ca"],
                    "jaccard_at_best": best["jaccard"], "ligand_heavy": manifest[entry]["ligand_heavy"],
                    "seed_model": top["model"], "seed_lig_rmsd": top["lig_rmsd"], "seed_cif": top["cif"]})
    with open(pb / "screen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    failures = [o for o in out if o["verdict"] == "clean_pose_failure"]
    with open(pb / "failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(failures)
    (pb / "failure_configs.txt").write_text(
        "".join(f"{pb / 'cofold_configs' / (o['entry'] + '.yaml')}\n" for o in failures))
    print({k: len(v) for k, v in verdicts.items()})
    print("clean pose failures:", [o["entry"] for o in failures])


if __name__ == "__main__":
    main()
