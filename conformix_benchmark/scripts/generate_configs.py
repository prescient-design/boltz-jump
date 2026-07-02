#!/usr/bin/env python3
"""
Generate boltz-jump YAML configs for all ConforMix benchmark proteins.

Usage:
    python scripts/generate_configs.py \
        --datasets_dir . \
        --configs_dir configs \
        --ref_pdbs_dir ref_pdbs
"""

import argparse
import csv
import os
from pathlib import Path


def load_sequences(seq_csv: Path) -> dict[str, str]:
    """Load sequences from a ConforMix sequences CSV."""
    seqs = {}
    with open(seq_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The key column is 'system_id' or 'test_case' depending on dataset
            key = row.get("system_id") or row.get("test_case")
            seqs[key] = row["sequence"].strip()
    return seqs


def write_yaml(out_path: Path, sequence: str, system_id: str):
    """Write a boltz-jump compatible YAML config."""
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: '1'
      sequence: {sequence}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(yaml_content)


def process_dataset(
    dataset_name: str,
    dataset_csv: Path,
    seq_csv: Path,
    configs_dir: Path,
):
    """Generate configs for one dataset."""
    sequences = load_sequences(seq_csv)
    out_dir = configs_dir / dataset_name

    with open(dataset_csv) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            system_id = row.get("system_id") or row.get("test_case")
            seq = sequences.get(system_id)
            if seq is None:
                print(f"  WARNING: no sequence for {system_id}, skipping")
                continue

            yaml_path = out_dir / f"{system_id}.yaml"
            write_yaml(yaml_path, seq, system_id)
            count += 1

    print(f"  {dataset_name}: wrote {count} configs to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=Path, default=Path("."))
    parser.add_argument("--configs_dir", type=Path, default=Path("configs"))
    args = parser.parse_args()

    datasets = [
        ("domainmotion", "domainmotion.csv", "domainmotion_sequences.csv"),
        ("membranetransporters", "membranetransporters.csv", "membranetransporters_sequences.csv"),
        ("crypticpockets", "crypticpockets.csv", "crypticpockets_sequences.csv"),
        ("foldswitching", "foldswitching.csv", "foldswitching_sequences.csv"),
    ]

    for name, ds_csv, seq_csv in datasets:
        ds_path = args.datasets_dir / ds_csv
        seq_path = args.datasets_dir / seq_csv
        if not ds_path.exists():
            print(f"  Skipping {name}: {ds_path} not found")
            continue
        process_dataset(name, ds_path, seq_path, args.configs_dir)


if __name__ == "__main__":
    main()
