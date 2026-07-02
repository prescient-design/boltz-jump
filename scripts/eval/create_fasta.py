#!/usr/bin/env env python3
"""
Convert YAML configuration files to a single FASTA file.
Extracts protein sequences from YAML files and writes them in FASTA format.
"""

import yaml
import argparse
from pathlib import Path
import os


def parse_yaml_file(yaml_path: Path) -> tuple[str, str]:
    """Parse a YAML file and extract protein information."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    assert len(data['sequences']) == 1, "Expected exactly one sequence per YAML file"

    protein = data['sequences'][0]['protein']
    sequence = protein['sequence']
    # Parse protein ID from the YAML filename
    protein_id = yaml_path.stem  # Filename without extension
    return protein_id, sequence

def write_fasta(sequences, output_path):
    """Write sequences to a FASTA file."""
    with open(output_path, 'w') as f:
        for protein_id, sequence in sequences:
            f.write(f'>{protein_id}\n')
            # Write sequence in lines of 80 characters (standard FASTA format)
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Convert YAML config files to a single FASTA file'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing YAML files'
    )
    parser.add_argument(
        'output_fasta',
        type=str,
        help='Output FASTA file path'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.yaml',
        help='File pattern to match (default: *.yaml)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Directory {input_dir} does not exist")
    
    # Find all YAML files
    yaml_files = sorted(list(input_dir.glob(args.pattern)))
    
    if not yaml_files:
        raise ValueError(f"No files matching pattern '{args.pattern}' found in {input_dir}")
    
    print(f"Found {len(yaml_files)} YAML file(s)")
    
    # Collect all sequences
    all_sequences = []
    for yaml_file in yaml_files:
        print(f"Processing {yaml_file.name}...")
        protein_id, sequence = parse_yaml_file(yaml_file)
        all_sequences.append((protein_id, sequence))
    
    if not all_sequences:
        raise ValueError("No sequences found in any YAML files")
    
    # Write to FASTA
    write_fasta(all_sequences, args.output_fasta)
    print(f"\nSuccessfully wrote {len(all_sequences)} sequence(s) to {os.path.abspath(args.output_fasta)}")


if __name__ == '__main__':
    main()