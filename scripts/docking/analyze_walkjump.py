#!/usr/bin/env python3
"""Score walk-jump pose-refinement trajectories and summarize by sigma.

Every frame of every run in <dataset>/walkjump_tasks.tsv is scored as sampled
against the crystal pose, and checked for valid ligand geometry (see common.py).
Writes, under $DOCKING_DIR/<dataset>/analysis/:

    frames.csv     one row per frame: ligand RMSD, pocket C-alpha RMSD, bond-length ratios, valid flag
    targets.csv    one row per (target, sigma): start RMSD, best valid-frame ligand RMSD over all
                   starts, seeds and frames, fraction of valid frames
    sigmas.csv     one row per sigma: targets rescued (best valid RMSD < 2 A), median best RMSD,
                   mean valid fraction

Usage:
    python scripts/docking/analyze_walkjump.py posebusters
    python scripts/docking/analyze_walkjump.py runs_n_poses
"""

import argparse
import csv
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import mdtraj as md
import numpy as np

from common import SUCCESS_RMSD_A, Ideal, PoseScorer, Reference, docking_dir

_IDEAL = {}


def target_info(dataset, root, name):
    """(target id, reference, SMILES) for a walk-jump start."""
    if dataset == "posebusters":
        target = name.rsplit("__", 1)[0]  # <entry>__start<k>
        entry = json.load(open(root / "manifest.json"))["entries"][target]
        ref = Reference.posebusters(docking_dir() / "data/posebusters/posebusters_benchmark_set" / target)
        return target, ref, entry["ligand_smiles"]
    system, start = name.rsplit("__", 1)
    row = next(r for r in csv.DictReader(open(root / "starts.csv")) if r["system"] == system and r["start"] == start)
    gt = docking_dir() / "data/runs_n_poses/ground_truth" / system
    return system, Reference.runs_n_poses(gt, row["crystal_ligand_chain"]), row["smiles"]


def analyze_run(task):
    dataset, root, cfg, sigma, seed, out_dir = task
    name = Path(cfg).stem
    pred = sorted(Path(out_dir).glob(f"boltz_jump_results_*_{name}/predictions/{name}"))
    if not pred or not (pred[0] / f"{name}_model_0.xtc").exists():
        return name, sigma, seed, None
    pred = pred[0]
    target, ref, smiles = target_info(dataset, root, name)
    if smiles not in _IDEAL:
        _IDEAL[smiles] = Ideal(smiles)
    top = pred / f"{name}_model_0.cif"
    scorer = PoseScorer(ref, top, _IDEAL[smiles])
    start = scorer.score(md.load(str(top)).xyz[0] * 10.0)
    traj = md.load(str(pred / f"{name}_model_0.xtc"), top=str(top))
    rows, poses = [], []
    for k in range(traj.n_frames):
        s = scorer.score(traj.xyz[k] * 10.0, idealize=True)
        pose = s.pop("ideal_xyz", None)
        rows.append({"target": target, "start": name, "sigma": sigma, "seed": seed, "frame": k,
                     "start_lig_rmsd": round(start["lig_rmsd"], 3), **{
                         key: (round(v, 3) if isinstance(v, float) else v) for key, v in s.items()}})
        poses.append(pose)
    return name, sigma, seed, rows, poses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["posebusters", "runs_n_poses"])
    parser.add_argument("--tasks_name", default="walkjump_tasks.tsv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = docking_dir() / args.dataset
    tasks = [(args.dataset, root, *line.rstrip("\n").split("\t")) for line in open(root / args.tasks_name)]
    with Pool(args.workers) as pool:
        results = pool.map(analyze_run, tasks, chunksize=1)
    missing = [(n, s, sd) for n, s, sd, rows, _ in results if rows is None]
    if missing:
        print(f"{len(missing)} runs missing, e.g. {missing[:5]}")
    frames, pose_of = [], {}
    for _, _, _, rows, poses in results:
        if not rows:
            continue
        for r, pose in zip(rows, poses):
            key = (r["target"], float(r["sigma"]), r["start"], r["frame"])
            if pose is not None:
                pose_of[key] = pose
            frames.append(r)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    fields = ["target", "start", "sigma", "seed", "frame", "start_lig_rmsd", "lig_rmsd", "pocket_ca", "jaccard",
              "diameter", "min_env_dist", "bond_ratio_min", "bond_ratio_max", "valid",
              "ideal_rmsd", "ideal_fit_rmsd"]
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frames)

    def medoid_rmsd(rs):
        """RMSD of the medoid pose: the valid pose closest to all the others.

        This is a label-free selector -- it uses only agreement between sampled
        poses, never the crystal structure -- so it measures what is actually
        deployable, unlike the oracle.
        """
        pts = [(r, pose_of.get((r["target"], float(r["sigma"]), r["start"], r["frame"])))
               for r in rs]
        pts = [(r, p) for r, p in pts if p is not None]
        if not pts:
            return None, None
        xyz = np.stack([p for _, p in pts])
        d = np.sqrt((((xyz[:, None] - xyz[None]) ** 2).sum(-1)).mean(-1))
        i = int(d.sum(1).argmin())
        return round(float(pts[i][0]["ideal_rmsd"]), 3), len(pts)

    by_target = defaultdict(list)
    for r in frames:
        by_target[(r["target"], float(r["sigma"]))].append(r)
    targets = []
    for (target, sigma), rs in sorted(by_target.items()):
        valid = [r for r in rs if r.get("valid") and r.get("ideal_rmsd") is not None]
        targets.append({"target": target, "sigma": sigma, "n_starts": len({r["start"] for r in rs}),
                        "n_frames": len(rs), "best_start_lig_rmsd": min(r["start_lig_rmsd"] for r in rs),
                        "best_raw_rmsd": min(r["lig_rmsd"] for r in rs),
                        "best_valid_rmsd": min(r["lig_rmsd"] for r in valid) if valid else None,
                        "best_ideal_rmsd": min(r["ideal_rmsd"] for r in valid) if valid else None,
                        "frac_valid": round(len(valid) / len(rs), 3),
                        "medoid_rmsd": medoid_rmsd(valid)[0]})
    with open(out / "targets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(targets[0]))
        w.writeheader()
        w.writerows(targets)

    by_sigma = defaultdict(list)
    for t in targets:
        by_sigma[t["sigma"]].append(t)
    n_targets = len({t["target"] for t in targets})
    sigmas = []
    for sigma, ts in sorted(by_sigma.items()):
        best = np.array([t["best_ideal_rmsd"] if t["best_ideal_rmsd"] is not None else np.inf for t in ts])
        med = np.array([t["medoid_rmsd"] if t["medoid_rmsd"] is not None else np.inf for t in ts])
        sigmas.append({"sigma": sigma, "n_targets": len(ts), "rescued": int((best < SUCCESS_RMSD_A).sum()),
                       "rescued_medoid": int((med < SUCCESS_RMSD_A).sum()),
                       "median_best_ideal_rmsd": round(float(np.median(best)), 2),
                       "mean_frac_valid": round(float(np.mean([t["frac_valid"] for t in ts])), 3)})
    with open(out / "sigmas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sigmas[0]))
        w.writeheader()
        w.writerows(sigmas)
    print(f"{n_targets} targets")
    for s in sigmas:
        print(s)


if __name__ == "__main__":
    main()
