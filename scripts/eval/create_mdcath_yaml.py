#!/usr/bin/env python3
"""
PDB to Boltz-2 Input Generator

Fetches protein sequences from PDB and generates Boltz-2 input files.
Example: python create_mdcath_yaml.py 1zxfA00
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


def get_chain_sequences(structure: StructureV2) -> dict[str, str]:
    chain_sequences = {}
    for chain in structure.chains:
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        chain_residues = structure.residues[res_start:res_end]
        residue_names = [residue["name"] for residue in chain_residues]
        for i, name in enumerate(residue_names):
            if name in {'HSD', 'HSE', 'HSP'}:
                residue_names[i] = 'HIS'  # Standardize histidine variants
        sequence = gemmi.one_letter_code(residue_names)
        # Check that no X (unknown residues) are present
        if 'X' in sequence:
            raise ValueError(f"Chain {chain['name']} contains unknown residues represented as 'X' in sequence: {sequence}")
        chain_sequences[chain["name"]] = sequence
    return chain_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Generate Boltz-2 input files from PDB structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_yaml.py 1zxfA00 --output-dir ./outputs
        """,
    )

    parser.add_argument(
        "--mdcath-dir", help="Directory for ATLAS data"
    )
    parser.add_argument(
        "--mdcath-msa-dir", help="Directory for ATLAS MSA data",
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
    pdb_id = args.input
    pdb_path = Path(args.mdcath_dir) / f"{args.input}.pdb"
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
    logging.info(f"Fetching sequences...")

    # Generate Boltz-2 input
    if args.mdcath_msa_dir:
        msa_path = Path(args.mdcath_msa_dir) / f"{args.input}.a3m"
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
