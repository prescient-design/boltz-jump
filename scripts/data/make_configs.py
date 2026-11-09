#!/usr/bin/env python3
"""Generate the benchmark input YAMLs under configs/.

Paths are written as ${VAR} placeholders (ATLAS_DIR, ATLAS_MSA_DIR, MDCATH_DIR,
MDCATH_MSA_DIR, RESULTS_DIR) that are resolved from .env at prediction time.

ATLAS and mdCATH (sequences from data/splits/{atlas,mdcath}_test_sequences.csv)
get one directory per MSA and initial-structure condition:

    {prefix}_msa                  precomputed MSA, experimental initial structure
    {prefix}_empty_msa            no MSA, experimental initial structure
    {prefix}_msa_no_init          precomputed MSA, Boltz-2 generates the initial structure
    {prefix}_empty_msa_no_init    no MSA, Boltz-2 generates the initial structure
    {prefix}_empty_msa_boltz2_init  no MSA, initial structure from the {prefix}_msa_no_init run

ConforMix (sequences from conformix_benchmark/{subset}_sequences.csv) gets
configs/conformix_benchmark/{subset}/, with MSAs from the MSA server.

Usage:
    python scripts/data/make_configs.py              # all datasets
    python scripts/data/make_configs.py atlas mdcath
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "atlas": {
        "prefix": "atlas_test",
        "sequences": "data/splits/atlas_test_sequences.csv",
        "msa": "${ATLAS_MSA_DIR}/{name}.a3m",
        "structure": "pdb: ${ATLAS_DIR}/{name}/protein/{name}.pdb",
    },
    "mdcath": {
        "prefix": "mdcath",
        "sequences": "data/splits/mdcath_test_sequences.csv",
        "msa": "${MDCATH_MSA_DIR}/{name}.a3m",
        "structure": "pdb: ${MDCATH_DIR}/{name}.pdb",
    },
}
CONFORMIX_SUBSETS = ["crypticpockets", "domainmotion", "foldswitching", "membranetransporters"]
# Initial structures for the boltz2_init configs: the seed written by the
# walk-jump run on the {prefix}_msa_no_init configs.
BOLTZ2_INIT = (
    "cif: ${{RESULTS_DIR}}/{prefix}_msa_no_init/boltz_jump_results_s=2.0_d=1.0_f=0.1_{name}"
    "/predictions/{name}/{name}_model_0.cif"
)


def config(chain_id, sequence, msa=None, structure=None):
    text = f"version: 1\nsequences:\n  - protein:\n      id: '{chain_id}'\n      sequence: {sequence}\n"
    if msa is not None:
        text += f"      msa: {msa}\n"
    text += "\n"
    if structure is not None:
        text += f"initial_structure:\n  {structure}\n"
    return text


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_benchmark(dataset, out_root):
    spec = DATASETS[dataset]
    prefix = spec["prefix"]
    with open(ROOT / spec["sequences"]) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        name, chain_id, seq = row["name"], row["chain_id"], row["sequence"]
        msa = spec["msa"].replace("{name}", name)
        structure = spec["structure"].replace("{name}", name)
        variants = {
            "msa": config(chain_id, seq, msa, structure),
            "empty_msa": config(chain_id, seq, "empty", structure),
            "msa_no_init": config(chain_id, seq, msa),
            "empty_msa_no_init": config(chain_id, seq, "empty"),
            "empty_msa_boltz2_init": config(chain_id, seq, "empty", BOLTZ2_INIT.format(prefix=prefix, name=name)),
        }
        for variant, text in variants.items():
            write(out_root / f"{prefix}_{variant}" / "configs" / f"{name}.yaml", text)
    print(f"{dataset}: {len(rows)} targets x {len(variants)} conditions")


def make_conformix(out_root):
    for subset in CONFORMIX_SUBSETS:
        with open(ROOT / "conformix_benchmark" / f"{subset}_sequences.csv") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            text = config("1", row["sequence"]).rstrip("\n") + "\n"
            write(out_root / "conformix_benchmark" / subset / f"{row.get('system_id') or row['test_case']}.yaml", text)
        print(f"conformix/{subset}: {len(rows)} targets")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", choices=[*DATASETS, "conformix"],
                        help="Datasets to generate (default: all)")
    parser.add_argument("--out_dir", default=str(ROOT / "configs"))
    args = parser.parse_args()
    out_root = Path(args.out_dir)
    for dataset in args.datasets or [*DATASETS, "conformix"]:
        if dataset == "conformix":
            make_conformix(out_root)
        else:
            make_benchmark(dataset, out_root)


if __name__ == "__main__":
    main()
