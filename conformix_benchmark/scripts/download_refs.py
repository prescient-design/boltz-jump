#!/usr/bin/env python3
"""
Download reference PDB structures for all ConforMix benchmark systems.

Usage:
    python scripts/download_refs.py --datasets_dir . --ref_pdbs_dir ref_pdbs
"""

import argparse
import csv
import os
import urllib.request
from pathlib import Path


def download_pdb(pdb_id: str, out_dir: Path):
    """Download a PDB file from RCSB. pdb_id is like '1CLL' (4 chars)."""
    pdb_code = pdb_id[:4].upper()
    out_path = out_dir / f"{pdb_code}.cif"
    if out_path.exists():
        return

    url = f"https://files.rcsb.org/download/{pdb_code}.cif"
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"  Downloaded {pdb_code}")
    except Exception as e:
        print(f"  FAILED to download {pdb_code}: {e}")


def collect_pdb_ids(datasets_dir: Path) -> set[str]:
    """Collect all unique PDB codes from all dataset CSVs."""
    pdb_ids = set()

    # Domain motion: pdb1, pdb2 columns (format: "1CLL_A")
    for csv_name, col_names in [
        ("domainmotion.csv", ["pdb1", "pdb2"]),
        ("crypticpockets.csv", ["pdb1", "pdb2"]),
        ("foldswitching.csv", ["pdb1", "pdb2"]),
    ]:
        csv_path = datasets_dir / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in col_names:
                    val = row.get(col, "").strip()
                    if val:
                        pdb_ids.add(val.split("_")[0].upper())

    # Membrane transporters: pdbid_if, pdbid_of columns
    csv_path = datasets_dir / "membranetransporters.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ["pdbid_if", "pdbid_of"]:
                    val = row.get(col, "").strip()
                    if val:
                        pdb_ids.add(val.split("_")[0].upper())

    return pdb_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=Path, default=Path("."))
    parser.add_argument("--ref_pdbs_dir", type=Path, default=Path("ref_pdbs"))
    args = parser.parse_args()

    args.ref_pdbs_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = collect_pdb_ids(args.datasets_dir)
    print(f"Found {len(pdb_ids)} unique PDB codes to download")

    for pdb_id in sorted(pdb_ids):
        download_pdb(pdb_id, args.ref_pdbs_dir)

    print(f"\nDone. Files in {args.ref_pdbs_dir}")


if __name__ == "__main__":
    main()
