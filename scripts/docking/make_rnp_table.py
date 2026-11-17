#!/usr/bin/env python3
"""Emit the Runs N' Poses LaTeX table and the numbers quoted in the text.

Reads $DOCKING_DIR/runs_n_poses/analysis/targets.csv (from analyze_walkjump.py)
and writes the table body plus a summary of every count the prose cites, so the
text and the table cannot drift apart.

Usage:
    python scripts/docking/make_rnp_table.py --sigma 4 --out rnp_table.tex
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from common import SUCCESS_RMSD_A, docking_dir

COLUMNS = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--out", default="rnp_table.tex")
    args = parser.parse_args()

    root = docking_dir() / "runs_n_poses"
    rows = [r for r in csv.DictReader(open(root / "analysis" / "targets.csv"))
            if float(r["sigma"]) == args.sigma and r["best_ideal_rmsd"] not in ("", "None")]
    heavy = {r["system"]: r["ligand_heavy"] for r in csv.DictReader(open(root / "targets.csv"))}
    n_total = len({r["system"] for r in csv.DictReader(open(root / "targets.csv"))})

    rows.sort(key=lambda r: float(r["best_ideal_rmsd"]))
    oracle = np.array([float(r["best_ideal_rmsd"]) for r in rows])
    medoid = np.array([float(r["medoid_rmsd"]) if r["medoid_rmsd"] not in ("", "None") else np.inf
                       for r in rows])
    seed = np.array([float(r["best_start_lig_rmsd"]) for r in rows])
    starts = sorted({int(r["n_starts"]) for r in rows})

    # table body: PDB / heavy atoms / oracle RMSD, filled column-major
    cells = []
    for r in rows:
        pdb = r["target"].split("_")[0]
        v = float(r["best_ideal_rmsd"])
        val = f"\\textbf{{{v:.2f}}}" if v <= SUCCESS_RMSD_A else f"{v:.2f}"
        cells.append(f"\\texttt{{{pdb}}} & {heavy.get(r['target'], '')} & {val}")
    per_col = -(-len(cells) // COLUMNS)
    lines = []
    for i in range(per_col):
        row = [cells[i + c * per_col] if i + c * per_col < len(cells) else " & & " for c in range(COLUMNS)]
        lines.append(" & ".join(row) + r" \\")
    Path(args.out).write_text("\n".join(lines) + "\n")

    def pct(n):
        return f"{n}/{len(rows)} ({100 * n / len(rows):.0f}\\%)"

    print(f"targets scored at sigma={args.sigma:g}: {len(rows)} of {n_total}; starts per target: {starts}")
    print(f"  co-folding alone (all are failures by construction): 0/{len(rows)}")
    print(f"  walk-jump ORACLE  <= {SUCCESS_RMSD_A} A: {pct(int((oracle <= SUCCESS_RMSD_A).sum()))}")
    print(f"  medoid SELECTOR   <= {SUCCESS_RMSD_A} A: {pct(int((medoid <= SUCCESS_RMSD_A).sum()))}")
    print(f"  median oracle RMSD {np.median(oracle):.2f} A; median seed RMSD {np.median(seed):.2f} A")
    print(f"  improved on the seed: {pct(int((oracle < seed).sum()))}")
    for lo, hi in [(0, 4), (4, 7), (7, np.inf)]:
        m = (seed >= lo) & (seed < hi)
        if m.sum():
            label = f"seed {lo}-{hi} A" if np.isfinite(hi) else f"seed >= {lo} A"
            print(f"  {label}: {int((oracle[m] <= SUCCESS_RMSD_A).sum())}/{int(m.sum())} rescued")
    print(f"\ntable body -> {args.out}")


if __name__ == "__main__":
    main()
