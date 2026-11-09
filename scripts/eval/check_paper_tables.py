#!/usr/bin/env python3
"""Check the conformational-ensemble tables in the paper against regenerated results.

Parses tab:atlas, tab:mdcath, tab:ablation and tab:noised-seed out of main.tex
and compares every cell with the summary_table.csv written by
scripts/eval/plot.py. Exits non-zero if any cell differs beyond rounding.

Usage:
    python scripts/eval/check_paper_tables.py --tex main.tex \
        --atlas results_plots/atlas/summary_table.csv \
        --mdcath results_plots/mdcath/summary_table.csv
"""

import argparse
import csv
import re
import sys

# Paper row label -> summary_table.csv column.
METRICS = {
    "lDDT Precision": "lDDT Precision",
    "lDDT Recall": "lDDT Recall",
    "lDDT Diversity": "lDDT Diversity",
    "Pairwise RMSD": "Pairwise RMSD rho",
    "Per-residue RMSF": "Per target RMSF rho",
    "RMWD": "RMWD",
    "PCA": "Joint PCA W2",
    "PC similarity": "PC sim > 0.5 %",
    "Weak contacts": "Weak contacts J",
    "Transient contacts": "Transient contacts J",
    "Exposed residue": "Exposed residue J",
    "Exposed MI": "Exposed MI matrix rho",
}

MAIN_COLUMNS = [
    "Boltz-2 Jump (MSA, δ=1)",
    "Boltz-2 Jump (MSA, δ=2, 100fr)",
    "Boltz-2 Full MSA",
    "Boltz-2 MSA Subsample 32",
    "Boltz-2 MSA Subsample 64",
    "AlphaFlow MD Base",
    "AlphaFlow MD Distilled",
    "BioEmu",
]
ABLATION_COLUMNS = [
    "Boltz-2 Jump (MSA, δ=1)",
    "Boltz-2 Jump (no MSA, δ=1)",
    "Boltz-2 Jump (MSA, no init, δ=1)",
    "Boltz-2 Jump (no MSA, Boltz-2 init, δ=1)",
    "Boltz-2 Jump (no MSA, no init, δ=1)",
]
NOISED_SEED_COLUMNS = [
    "Boltz-2 Jump (MSA, δ=1)",
    "Noised Seed SDE",
    "Noised Seed ODE",
    "Noised Seed (1-step)",
]

# (label, columns, dataset for tables without row groups, number of leading
#  Boltz-2-variant columns marked gray when best among them)
TABLES = [
    ("tab:atlas", MAIN_COLUMNS, "atlas", 5),
    ("tab:mdcath", MAIN_COLUMNS, "mdcath", 5),
    ("tab:ablation", ABLATION_COLUMNS, None, 0),
    ("tab:noised-seed", NOISED_SEED_COLUMNS, None, 0),
]


def load_summary(path):
    return {row["path"]: row for row in csv.DictReader(open(path))}


def strip_latex(cell):
    cell = re.sub(r"\\(textbf|textcolor\{gray\})\{([^{}]*)\}", r"\2", cell)
    cell = re.sub(r"\\(textbf|textcolor\{gray\})\{([^{}]*)\}", r"\2", cell)
    return cell.strip()


def table_rows(tex, label):
    """Yield (dataset_group, metric_label, [cells]) for each data row of a table."""
    start = tex.index(f"\\label{{{label}}}")
    body = tex[start:tex.index("\\bottomrule", start)]
    body = body[body.index("\\midrule"):]
    group = None
    for line in body.split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if not line:
            continue
        if "\\multicolumn" in line:
            group = "mdcath" if "mdCATH" in line else "atlas"
            continue
        cells = [c.strip() for c in line.split("&")]
        yield group, cells[0], cells[1:]


def apply_fixes(tex, fixes):
    """Rewrite only the offending data rows, scoped to their table and row group."""
    lines = tex.split("\n")
    for label, dataset, row_label, _, fixed in fixes:
        start = next(i for i, l in enumerate(lines) if f"\\label{{{label}}}" in l)
        end = next(i for i in range(start, len(lines)) if "\\bottomrule" in lines[i])
        group = None
        for i in range(start, end):
            if "\\multicolumn" in lines[i]:
                group = "mdcath" if "mdCATH" in lines[i] else "atlas"
            head = lines[i].split("&")[0].strip()
            if head == row_label and (group is None or group == dataset):
                lines[i] = f"{row_label} & {' & '.join(fixed)} \\\\"
                break
        else:
            sys.exit(f"Could not locate row '{row_label}' in {label} ({dataset})")
    return "\n".join(lines)


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def expected_cells(row_label, displayed, n_gray):
    """Re-derive bold (best overall) and gray (best of the first n_gray columns) markup."""
    lower_better = "downarrow" in row_label
    best = min if lower_better else max
    vals = [float(v) if is_num(v) else None for v in displayed]
    valid = [v for v in vals if v is not None]
    top = best(valid)
    top_b2 = best([v for v in vals[:n_gray] if v is not None]) if n_gray else None
    out = []
    for i, (s, v) in enumerate(zip(displayed, vals)):
        if v is not None and v == top:
            s = f"\\textbf{{{s}}}"
        if n_gray and i < n_gray and v is not None and v == top_b2:
            s = f"\\textcolor{{gray}}{{{s}}}"
        out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex", default="main.tex")
    parser.add_argument("--atlas", default="results_plots/atlas/summary_table.csv")
    parser.add_argument("--mdcath", default="results_plots/mdcath/summary_table.csv")
    parser.add_argument("--fix", action="store_true",
                        help="Rewrite differing table rows in --tex in place (table rows only)")
    parser.add_argument("--print_fixed", action="store_true",
                        help="Print corrected LaTeX rows for every row that differs")
    args = parser.parse_args()

    tex = open(args.tex).read()
    summaries = {"atlas": load_summary(args.atlas), "mdcath": load_summary(args.mdcath)}

    n_checked, mismatches, bad_markup = 0, [], []
    for label, columns, fixed_dataset, n_gray in TABLES:
        for group, row_label, cells in table_rows(tex, label):
            dataset = fixed_dataset or group
            metric = next(v for k, v in METRICS.items() if row_label.startswith(k))
            displayed = []
            for model, cell in zip(columns, cells):
                paper = strip_latex(cell)
                actual = summaries[dataset][model][metric]
                n_checked += 1
                if actual in ("", "nan"):
                    displayed.append("---")
                    if paper not in ("---", ""):
                        mismatches.append((label, dataset, row_label, model, paper, actual))
                    continue
                d = decimals(paper) if is_num(paper) else 3
                displayed.append(f"{float(actual):.{d}f}")
                if not is_num(paper) or abs(float(paper) - float(actual)) > 0.5 * 10 ** -d + 1e-9:
                    mismatches.append((label, dataset, row_label, model, paper, actual))
            fixed = expected_cells(row_label, displayed, n_gray)
            if [c.replace(" ", "") for c in fixed] != [c.replace(" ", "") for c in cells]:
                bad_markup.append((label, dataset, row_label, cells, fixed))

    for label, dataset, row_label, model, paper, actual in mismatches:
        print(f"MISMATCH {label:16} {dataset:7} {row_label[:28]:28} {model:42} paper={paper:>7} actual={actual}")
    print(f"{n_checked - len(mismatches)}/{n_checked} table values match.")
    print(f"{len(bad_markup)} rows have a value or bold/gray markup that differs from the data.")
    if args.print_fixed:
        for label, dataset, row_label, cells, fixed in bad_markup:
            print(f"\n% {label} ({dataset})\n%  was: {row_label} & {' & '.join(cells)} \\\\")
            print(f"{row_label} & {' & '.join(fixed)} \\\\")
    if args.fix and bad_markup:
        open(args.tex, "w").write(apply_fixes(tex, bad_markup))
        print(f"Rewrote {len(bad_markup)} rows in {args.tex}.")
        return
    sys.exit(1 if mismatches or bad_markup else 0)


if __name__ == "__main__":
    main()
