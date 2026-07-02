#!/usr/bin/env python3
"""
Evaluate conformational coverage on ConforMix benchmarks using TMalign.

For each protein, computes min-RMSD and max-TM-score of generated samples
to each reference PDB structure. Reports coverage metrics matching the
ConforMix paper.

Usage:
    python scripts/eval_coverage.py \
        --dataset domainmotion \
        --datasets_dir . \
        --predictions_dir results/domainmotion \
        --ref_pdbs_dir ref_pdbs \
        --output_dir eval_results

Requires: gemmi, numpy, pandas, TMalign (in PATH)
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd


def load_dataset(datasets_dir: Path, dataset: str):
    """Load dataset CSV and return list of dicts with system info."""
    csv_path = datasets_dir / f"{dataset}.csv"
    systems = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            system_id = row.get("system_id") or row.get("test_case")

            if dataset == "membranetransporters":
                pdb1 = row["pdbid_if"]
                pdb2 = row["pdbid_of"]
            else:
                pdb1 = row["pdb1"]
                pdb2 = row["pdb2"]

            ref_rmsd = float(row["RMSD"])
            systems.append({
                "system_id": system_id,
                "pdb1": pdb1,
                "pdb2": pdb2,
                "ref_rmsd": ref_rmsd,
            })
    return systems


def extract_chain(in_path: Path, out_path: Path, chain_id: str = None):
    """Extract a single chain from CIF/PDB, write as CIF."""
    st = gemmi.read_structure(str(in_path))
    model = st[0]

    if chain_id:
        chains_to_remove = [c.name for c in model if c.name != chain_id]
        for name in chains_to_remove:
            model.remove_chain(name)

    st.make_mmcif_document().write_file(str(out_path))


def run_tmalign(pdb1: Path, pdb2: Path) -> dict | None:
    """Run TMalign and parse RMSD and TM-score."""
    try:
        result = subprocess.run(
            ["TMalign", str(pdb1), str(pdb2)],
            capture_output=True, text=True, timeout=60
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    output = result.stdout

    rmsd = None
    tm1 = None
    tm2 = None

    for line in output.split("\n"):
        if "RMSD=" in line and "Aligned length" in line:
            m = re.search(r"RMSD=\s*([\d.]+)", line)
            if m:
                rmsd = float(m.group(1))

        if line.startswith("TM-score="):
            m = re.search(r"TM-score=\s*([\d.]+)", line)
            if m:
                score = float(m.group(1))
                if tm1 is None:
                    tm1 = score
                else:
                    tm2 = score

    if rmsd is None:
        return None

    return {
        "rmsd": rmsd,
        "tm_score_1": tm1,  # normalized by prediction length
        "tm_score_2": tm2,  # normalized by reference length
        "tm_score": tm2 or 0,  # use reference-normalized (standard convention)
    }


def find_prediction_dirs(pred_dir: Path, system_id: str) -> list[Path]:
    """Find boltz-jump output directory for a system."""
    pattern = pred_dir / f"boltz_jump_results_*_{system_id}" / "predictions" / system_id
    matches = glob.glob(str(pattern))
    return [Path(d) for d in matches if Path(d).is_dir()]


def extract_frames_as_cifs(pred_dir: Path, system_id: str, tmp_dir: Path) -> list[Path]:
    """Load XTC trajectory and write each frame as CIF using gemmi."""
    import mdtraj

    cif_files = glob.glob(str(pred_dir / "*.cif"))
    xtc_files = [f for f in glob.glob(str(pred_dir / "*.xtc"))
                 if not f.endswith("_y.xtc")]

    frame_paths = []

    if xtc_files and cif_files:
        top_file = cif_files[0]
        template_st = gemmi.read_structure(top_file)

        for xtc_file in xtc_files:
            try:
                traj = mdtraj.load(xtc_file, top=top_file)
            except Exception as e:
                print(f"    Error loading XTC {xtc_file}: {e}")
                continue

            for frame_idx in range(traj.n_frames):
                coords = traj.xyz[frame_idx] * 10  # nm -> Angstrom
                st = template_st.clone()
                atom_idx = 0
                for model in st:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                if atom_idx < len(coords):
                                    atom.pos = gemmi.Position(
                                        float(coords[atom_idx][0]),
                                        float(coords[atom_idx][1]),
                                        float(coords[atom_idx][2]))
                                    atom_idx += 1

                frame_cif = tmp_dir / f"{system_id}_frame_{frame_idx}.cif"
                st.make_mmcif_document().write_file(str(frame_cif))
                frame_paths.append(frame_cif)

            print(f"    Loaded {traj.n_frames} frames from XTC")

    elif cif_files:
        # Fall back: just use CIF files directly (no need to copy)
        for cif_file in cif_files:
            frame_paths.append(Path(cif_file))
        print(f"    Using {len(frame_paths)} CIF files directly (no XTC found)")

    return frame_paths

    return pdb_paths


def evaluate_system(
    system: dict,
    pred_dir: Path,
    ref_pdbs_dir: Path,
    tmp_dir: Path,
) -> dict | None:
    """Evaluate one system using TMalign."""
    system_id = system["system_id"]

    # Find prediction directory
    pred_dirs = find_prediction_dirs(pred_dir, system_id)
    if not pred_dirs:
        print(f"  No predictions found for {system_id}")
        return None

    # Prepare reference PDBs
    pdb1_code, pdb1_chain = system["pdb1"].split("_")
    pdb2_code, pdb2_chain = system["pdb2"].split("_")

    ref1_cif = ref_pdbs_dir / f"{pdb1_code.upper()}.cif"
    ref2_cif = ref_pdbs_dir / f"{pdb2_code.upper()}.cif"

    if not ref1_cif.exists() or not ref2_cif.exists():
        print(f"  Missing ref PDBs for {system_id}")
        return None

    ref1_pdb = tmp_dir / f"{system_id}_ref1.cif"
    ref2_pdb = tmp_dir / f"{system_id}_ref2.cif"

    try:
        extract_chain(ref1_cif, ref1_pdb, pdb1_chain)
        extract_chain(ref2_cif, ref2_pdb, pdb2_chain)
    except Exception as e:
        print(f"  Error preparing refs for {system_id}: {e}")
        return None

    # Extract all frames from trajectory
    frame_files = extract_frames_as_cifs(pred_dirs[0], system_id, tmp_dir)
    if not frame_files:
        print(f"  No frames extracted for {system_id}")
        return None

    min_rmsd_to_ref1 = float("inf")
    min_rmsd_to_ref2 = float("inf")
    max_tm_to_ref1 = 0.0
    max_tm_to_ref2 = 0.0

    for pred_file in frame_files:
        result1 = run_tmalign(pred_file, ref1_pdb)
        if result1:
            min_rmsd_to_ref1 = min(min_rmsd_to_ref1, result1["rmsd"])
            max_tm_to_ref1 = max(max_tm_to_ref1, result1["tm_score"])

        result2 = run_tmalign(pred_file, ref2_pdb)
        if result2:
            min_rmsd_to_ref2 = min(min_rmsd_to_ref2, result2["rmsd"])
            max_tm_to_ref2 = max(max_tm_to_ref2, result2["tm_score"])

    if min_rmsd_to_ref1 == float("inf") or min_rmsd_to_ref2 == float("inf"):
        print(f"  TMalign failed for {system_id}")
        return None

    best_rmsd = min(min_rmsd_to_ref1, min_rmsd_to_ref2)
    worst_rmsd = max(min_rmsd_to_ref1, min_rmsd_to_ref2)

    return {
        "system_id": system_id,
        "rmsd_to_ref1": min_rmsd_to_ref1,
        "rmsd_to_ref2": min_rmsd_to_ref2,
        "tm_to_ref1": max_tm_to_ref1,
        "tm_to_ref2": max_tm_to_ref2,
        "best_match_rmsd": best_rmsd,
        "worst_match_rmsd": worst_rmsd,
        "best_match_tm": max(max_tm_to_ref1, max_tm_to_ref2),
        "worst_match_tm": min(max_tm_to_ref1, max_tm_to_ref2),
        "ref_rmsd": system["ref_rmsd"],
        "best_match_frac": best_rmsd / system["ref_rmsd"],
        "worst_match_frac": worst_rmsd / system["ref_rmsd"],
        "n_predictions": len(frame_files),
    }


def compute_coverage(results: list[dict], key: str, thresholds: list[float]) -> dict:
    """Coverage at X = fraction of proteins where <key> <= X."""
    fracs = [r[key] for r in results if r is not None]
    n = len(fracs)
    if n == 0:
        return {}
    return {t: sum(1 for f in fracs if f <= t) / n for t in thresholds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["domainmotion", "membranetransporters",
                                 "crypticpockets", "foldswitching"])
    parser.add_argument("--datasets_dir", type=Path, default=Path("."))
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--ref_pdbs_dir", type=Path, default=Path("ref_pdbs"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    systems = load_dataset(args.datasets_dir, args.dataset)
    print(f"Loaded {len(systems)} systems from {args.dataset}")

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for system in systems:
            print(f"Evaluating {system['system_id']}...")
            result = evaluate_system(
                system, args.predictions_dir, args.ref_pdbs_dir, tmp_path
            )
            if result:
                results.append(result)
                print(f"  RMSD: best={result['best_match_rmsd']:.2f}A, "
                      f"worst={result['worst_match_rmsd']:.2f}A "
                      f"(ref={result['ref_rmsd']:.2f}A)")
                print(f"  TM:   best={result['best_match_tm']:.3f}, "
                      f"worst={result['worst_match_tm']:.3f}")

    if not results:
        print("No results to report!")
        return

    df = pd.DataFrame(results)
    csv_out = args.output_dir / f"{args.dataset}_results.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nPer-protein results saved to {csv_out}")

    thresholds = [0.1 * i for i in range(1, 11)]

    best_coverage = compute_coverage(results, "best_match_frac", thresholds)
    worst_coverage = compute_coverage(results, "worst_match_frac", thresholds)

    print(f"\n{'='*60}")
    print(f"COVERAGE RESULTS: {args.dataset} ({len(results)} proteins)")
    print(f"{'='*60}")
    print(f"\n{'Threshold':>12} {'Best-matched':>14} {'Worst-matched':>15}")
    print(f"{'-'*12:>12} {'-'*14:>14} {'-'*15:>15}")
    for t in thresholds:
        b = best_coverage.get(t, 0)
        w = worst_coverage.get(t, 0)
        print(f"{t:>11.0%} {b:>14.3f} {w:>15.3f}")

    print(f"\n** Coverage at 50% (ConforMix Table 1 metric) **")
    print(f"  Best-matched:  {best_coverage.get(0.5, 0):.3f}")
    print(f"  Worst-matched: {worst_coverage.get(0.5, 0):.3f}")

    mean_best_tm = np.mean([r["best_match_tm"] for r in results])
    mean_worst_tm = np.mean([r["worst_match_tm"] for r in results])
    print(f"\n** Mean TM-scores **")
    print(f"  Best-matched:  {mean_best_tm:.3f}")
    print(f"  Worst-matched: {mean_worst_tm:.3f}")

    summary = {
        "dataset": args.dataset,
        "n_proteins": len(results),
        "best_coverage": {str(k): v for k, v in best_coverage.items()},
        "worst_coverage": {str(k): v for k, v in worst_coverage.items()},
        "mean_best_tm": float(mean_best_tm),
        "mean_worst_tm": float(mean_worst_tm),
    }
    summary_path = args.output_dir / f"{args.dataset}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()