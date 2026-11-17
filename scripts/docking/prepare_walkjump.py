#!/usr/bin/env python3
"""Write walk-jump configs and the SLURM task list for pose refinement.

PoseBusters: one start per clean pose failure (failures.csv from
score_cofold.py), the top-ranked co-folded sample.
Runs N' Poses: up to --n_starts starts per target (starts.csv from
prepare_runs_n_poses.py), chosen by farthest-point sampling of the
pocket-aligned ligand centroid among the authors' Boltz-2 samples.

Each start is run at every sigma in --sigmas with --seeds walk-jump seeds.
Writes <dataset>/walkjump_configs/*.yaml and <dataset>/walkjump_tasks.tsv
(config, sigma, seed, output dir), the input of scripts/slurm/docking_walkjump.sh.

Usage:
    python scripts/docking/prepare_walkjump.py posebusters
    python scripts/docking/prepare_walkjump.py runs_n_poses --sigmas 4
"""

import argparse
import csv
from pathlib import Path

from common import docking_dir

DEFAULT_SIGMAS = [1, 2, 3, 4, 6, 8, 12, 16]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["posebusters", "runs_n_poses"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n_starts", type=int, default=5, help="posebusters: co-folded samples used as starts")
    parser.add_argument("--tasks_name", default="walkjump_tasks.tsv")
    args = parser.parse_args()

    root = docking_dir() / args.dataset
    cfg_dir = root / "walkjump_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "posebusters":
        # Every co-folded sample of a failure is a start (5 per entry), ranked by confidence, so
        # the protocol matches Runs N' Poses' diverse starts.
        failures = {r["entry"] for r in csv.DictReader(open(root / "failures.csv"))}
        samples = {}
        for r in csv.DictReader(open(root / "cofold_scores.csv")):
            if r["entry"] in failures:
                samples.setdefault(r["entry"], []).append(r)
        starts = []
        for entry, rows in sorted(samples.items()):
            rows.sort(key=lambda r: -float(r["confidence_score"] or 0.0))
            for k, r in enumerate(rows[:args.n_starts]):
                starts.append({"name": f"{entry}__start{k}", "cofold_config": root / "cofold_configs" / f"{entry}.yaml",
                               "cif": r["cif"]})
    else:
        starts = [{"name": f"{r['system']}__{r['start']}", "cofold_config": Path(r["config"]), "cif": r["cif"]}
                  for r in csv.DictReader(open(root / "starts.csv"))]

    tasks = []
    for s in starts:
        text = Path(s["cofold_config"]).read_text().rstrip("\n")
        cfg = cfg_dir / f"{s['name']}.yaml"
        cfg.write_text(f"{text}\n\ninitial_structure:\n  cif: {s['cif']}\n")
        for sigma in args.sigmas:
            sig = f"{sigma:g}"
            for seed in range(args.seeds):
                tasks.append(f"{cfg}\t{sig}\t{seed}\t{root / 'walkjump' / s['name'] / f'sigma{sig}' / f'seed{seed}'}\n")
    (root / args.tasks_name).write_text("".join(tasks))
    print(f"{len(starts)} starts x {len(args.sigmas)} sigmas x {args.seeds} seeds = {len(tasks)} runs "
          f"-> {root / args.tasks_name}")


if __name__ == "__main__":
    main()
