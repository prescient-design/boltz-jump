#!/usr/bin/env python3
"""Stage the data archive that accompanies the paper.

Collects exactly the ensembles behind the published tables, plus the MSAs
needed to regenerate them, into one directory tree with a manifest and
checksums. Nothing is uploaded; review the staged tree, then upload it.

Three of the published columns are backed by corrected re-runs (the `_v2`
directories): the "no MSA, Boltz-2 init" ablation column, which was originally
seeded from the wrong structure, and the noised-seed SDE/ODE baselines, which
were originally run with a noise-scale bug. The archive carries the corrected
data under the paper's own column names.

Usage:
    python scripts/data/stage_zenodo.py --out_dir /path/to/staging
"""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
import os

ROOT = Path(__file__).resolve().parents[2]

# paper column -> directory under results_organized/<dataset>/<target>/
MODELS = {
    "Boltz-2 Jump (MSA, delta=1)": "boltz2_jump_msa_d1",
    "Boltz-2 Jump (MSA, delta=2, 100 frames)": "boltz2_jump_msa_d2_100",
    "Boltz-2 Jump (no MSA, delta=1)": "boltz2_jump_empty_msa_d1",
    "Boltz-2 Jump (MSA, no init, delta=1)": "boltz2_jump_msa_no_init_d1",
    "Boltz-2 Jump (no MSA, no init, delta=1)": "boltz2_jump_empty_msa_no_init_d1",
    "Boltz-2 Jump (no MSA, Boltz-2 init, delta=1)": "boltz2_jump_empty_msa_boltz2_init_d1_v2",
    "Boltz-2 Full MSA": "boltz2_full_msa",
    "Boltz-2 MSA Subsample 32": "boltz2_msa_subsample_32",
    "Boltz-2 MSA Subsample 64": "boltz2_msa_subsample_64",
    "Noised Seed SDE": "boltz2_noised_seed_sde_v2",
    "Noised Seed ODE": "boltz2_noised_seed_ode_v2",
    "Noised Seed (1-step)": "boltz2_noised_seed_one_step",
    "AlphaFlow MD Base": "alphaflow_md_base",
    "AlphaFlow MD Distilled": "alphaflow_md_distilled",
    "BioEmu": "bioemu",
}

DATASETS = {"atlas": ("atlas_test_msa", "atlas_test.txt", "ATLAS_MSA_DIR"),
            "mdcath": ("mdcath_msa", "mdcath_test.txt", "MDCATH_MSA_DIR")}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--organized", default=str(ROOT / "results_organized"))
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True), override=False)
    out = Path(args.out_dir)
    organized = Path(args.organized)
    rows, missing = [], []

    for dataset, (subdir, split_file, msa_var) in DATASETS.items():
        targets = [t.strip() for t in open(ROOT / "github/data/splits" / split_file) if t.strip()]
        targets = [t.split("_320")[0] for t in targets]  # mdCATH split carries a temperature suffix
        for target in targets:
            for label, model in MODELS.items():
                src = organized / subdir / target / model
                dst = out / "ensembles" / dataset / target / model
                if not src.is_dir():
                    missing.append(f"{dataset}/{target}/{model}")
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                for name in ("traj.xtc", "system.cif"):
                    if (src / name).exists():
                        shutil.copy2(src / name, dst / name)
                        rows.append({"dataset": dataset, "target": target, "model": model,
                                     "paper_column": label, "file": f"ensembles/{dataset}/{target}/{model}/{name}"})
        # MSAs: only the evaluation targets, not the whole collection
        msa_dir = Path(os.environ[msa_var])
        for target in targets:
            src = msa_dir / f"{target}.a3m"
            if src.exists():
                dst = out / "msa" / dataset
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / src.name)
                rows.append({"dataset": dataset, "target": target, "model": "", "paper_column": "MSA",
                             "file": f"msa/{dataset}/{src.name}"})

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "target", "model", "paper_column", "file"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "SHA256SUMS", "w") as f:
        for r in sorted(rows, key=lambda r: r["file"]):
            f.write(f"{sha256(out / r['file'])}  {r['file']}\n")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"staged {len(rows)} files, {size / 2**30:.2f} GiB -> {out}")
    print(f"  {len(MODELS)} models x {sum(len(list((out / 'ensembles' / d).iterdir())) for d in DATASETS if (out / 'ensembles' / d).is_dir())} targets")
    if missing:
        print(f"  MISSING {len(missing)} model directories, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
