#!/usr/bin/env python3
"""
PDB to Boltz-2 Input Generator

Fetches protein sequences from PDB and generates Boltz-2 input files.
Example: python create_atlas_yaml.py 1UBQ_A
"""
import mdtraj as md
import sys
import requests
import argparse
import re
import gemmi
from pathlib import Path
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s]: %(message)s")

from boltz.data.types import StructureV2
from boltz.data.parse.pdb import parse_pdb


def generate_boltz_config(chain_sequences: dict[str, str], pdb_path: Path, msa_path: Optional[str] = None) -> str:
    """
    Generate Boltz-2 input file content.
    
    Args:
        chain_sequences (dict): Mapping of chain IDs to their sequences
        pdb_path (Path): Path to the PDB/CIF file
        
    Returns:
        str: Boltz-2 input file content
    """
    boltz_config = "version: 1\n"

    # Build the sequences section
    sequences_section = "sequences:\n"
    for chain_id, sequence in chain_sequences.items():
        sequences_section += f"  - protein:\n"
        sequences_section += f"      id: '{str(chain_id)}'\n"
        sequences_section += f"      sequence: {sequence}\n"
        if msa_path is not None:
            sequences_section += f"      msa: {msa_path}\n"
    
    # Build the complete input
    boltz_config += sequences_section
    boltz_config += f"\ninitial_structure:\n"
    boltz_config += f"  pdb: {pdb_path}\n"
    
    return boltz_config


def parse_pdb_chain_input(input_str: str) -> tuple[str, str]:
    """
    Parse input that could be either 'PDB_ID' or 'PDB_ID_CHAIN' format.

    Args:
        input_str (str): Input string like '1UBQ' or '1UBQ_A'

    Returns:
        tuple: (pdb_id, chain_id or None)
    """
    input_str = input_str.strip().upper()

    if "_" not in input_str:
        raise ValueError("Input should be in PDB_ID_CHAIN format (e.g., 1UBQ_A)")

    parts = input_str.split("_")
    if len(parts) != 2:
        raise ValueError("Format should be PDB_ID_CHAIN")
    return parts[0], parts[1]


def get_chain_sequences(structure: StructureV2) -> dict[str, str]:
    chain_sequences = {}
    for chain in structure.chains:
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        chain_residues = structure.residues[res_start:res_end]
        sequence = gemmi.one_letter_code([residue["name"] for residue in chain_residues])
        chain_sequences[chain["name"]] = sequence
    return chain_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Generate Boltz-2 input files from PDB structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_yaml.py 1UBQ_A --output-dir ./outputs
        """,
    )

    parser.add_argument(
        "--atlas-dir", help="Directory for ATLAS data"
    )
    parser.add_argument(
        "--atlas-msa-dir", help="Directory for ATLAS MSA data",
        default=None,
    )
    parser.add_argument(
        "input", help="PDB_CHAIN format (e.g., 1UBQ_A)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for .cif and .yaml files",
    )

    args = parser.parse_args()

    # Parse PDB and chain information
    pdb_id, chain_id = parse_pdb_chain_input(args.input)

    pdb_path = Path(args.atlas_dir) / args.input / "protein" / f"{args.input}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found at {pdb_path}")

    # Parse the PDB file to get the structure
    cache = "~/.boltz"
    cache = Path(cache).expanduser()
    moldir = cache / "mols"
    assert moldir.exists(), f"moldir does not exist: {moldir}"
    structure = parse_pdb(pdb_path, moldir=moldir, use_assembly=False, compute_interfaces=False).data

    # Extract sequences for each chain
    chain_sequences = get_chain_sequences(structure)
    logging.info(f"Fetching sequence for {pdb_id} chain {chain_id}...")

    # Generate Boltz-2 input
    if args.atlas_msa_dir:
        msa_path = Path(args.atlas_msa_dir) / f"{args.input}.a3m"
        assert msa_path.exists(), f"MSA file not found at {msa_path}"
    else:
        msa_path = None
    boltz_config = generate_boltz_config(chain_sequences, pdb_path, msa_path=msa_path)

    # Output result
    output_path = Path(args.output_dir) / "configs" / f"{args.input}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(boltz_config)
    logging.info(f"Boltz-2 input YAML saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
