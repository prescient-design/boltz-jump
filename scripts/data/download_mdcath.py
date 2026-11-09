#!/usr/bin/env python3
"""
Download and convert specific mdCATH dataset entries from HuggingFace.

This script downloads H5 files for specified CATH domain entries from the
mdCATH dataset and converts them to XTC/PDB format for the 320K temperature
simulations.

Based on mdCATH conversion utilities from:
https://github.com/compsciencelab/mdCATH/blob/main/user-utils/convert_mdCATH.py

Usage:
    python download_and_convert_mdcath.py [--output-dir OUTPUT_DIR] [--convert] [--temp-list TEMP ...] [--replica-list REPLICA ...]
"""

import os
import argparse
import h5py
import tempfile
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import mdtraj as md



# List of entries to download
ENTRIES = [
    "4i69A00_320",
    "2dckA02_320",
    "4i6uB00_320",
    "2kk2A00_320",
    "3fg6A02_320",
    "3ufbA01_320",
    "1wwjA00_320",
    "1a87A02_320",
    "4jmfB00_320",
    "4pt1B00_320",
    "3nv0A00_320",
    "2hg7A00_320",
    "2v0rA00_320",
    "1f8yA00_320",
    "2gzoA02_320",
    "1lddA00_320",
    "2hgkA01_320",
    "1uf2C02_320",
    "4r5qA00_320",
    "3i3lA02_320",
    "2oudA00_320",
    "1q6aA00_320",
    "1d2dA00_320",
    "3r5dA01_320",
    "3jciA00_320",
    "2k5iA02_320",
    "3qiiA00_320",
    "2k54A00_320",
    "2kc3A00_320",
    "3i32A02_320",
    "1wosA04_320",
    "2kkmA01_320",
    "1zxfA00_320",
    "4b9bA02_320",
    "3c6fA01_320",
    "2kkuA00_320",
    "1d0bA00_320",
    "2a7kB01_320",
    "3wy8A01_320",
    "4fhdA02_320",
]

REPO_ID = "compsciencelab/mdCATH"


# ============================================================================
# Conversion functions from mdCATH user-utils/convert_mdCATH.py
# ============================================================================

def _open_h5_file(h5):
    if isinstance(h5, str):
        h5 = h5py.File(h5, "r")
    code = [_ for _ in h5][0]
    return h5, code


def _extract_structure(h5, code):
    """
    Extracts the protein structure in PDB format from an H5 file.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    code : str
        The identifier for the dataset in the H5 file.

    Returns:
    str
        The filename of the temporary PDB file containing the protein structure.
    """
    pdb = h5[code]["pdbProteinAtoms"][()]
    
    # Decode and modify PDB content to replace non-standard histidine names
    if isinstance(pdb, bytes):
        pdb_text = pdb.decode('utf-8')
    else:
        pdb_text = pdb
    
    # Process line by line to handle column-specific replacements
    lines = pdb_text.split('\n')
    modified_lines = []
    
    for line in lines:
        if line.startswith(('ATOM', 'HETATM')) and len(line) >= 21:
            # Replace histidine variants in residue name (columns 17-20)
            res_name = line[17:20] if len(line) > 20 else ''
            if res_name in ['HSP', 'HSE', 'HSD']:
                line = line[:17] + 'HIS' + line[20:]
            
            # Replace chain ID P0 with A (columns 21-22)
            # Note: PDB format has chain at column 21 (0-indexed: 21)
            if len(line) > 22 and line[21:23] == 'P0':
                line = line[:21] + 'A ' + line[23:]
        
        modified_lines.append(line)
    
    pdb_text = '\n'.join(modified_lines)

    # Write modified PDB to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode='w') as pdbfile:
        pdbfile.write(pdb_text)
        pdbfile.flush()
        pdb_file_name = pdbfile.name
    return pdb_file_name


def _extract_structure_and_coordinates(h5, code, temp, replica):
    """
    Extracts the structure in PDB format and coordinates from an H5 file based on temperature and replica.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    code : str
        The identifier for the dataset in the H5 file.
    temp : int or float
        The temperature (in Kelvin).
    replica : int
        The replica number.

    Returns:
    tuple
        A tuple containing the PDB data as bytes and the coordinates as a numpy array.
    """
    pdb_file_name = _extract_structure(h5, code)
    coords = h5[code][f"{temp}"][f"{replica}"]["coords"][:]
    coords = coords / 10.0
    
    return pdb_file_name, coords


def convert_to_mdtraj(h5, temp, replica):
    """
    Convert data from an H5 file to an MDTraj trajectory object.

    This function extracts the first protein atom structure and coordinates
    for a given temperature and replica from an H5 file and creates an MDTraj
    trajectory object. This object can be used for further molecular dynamics
    analysis.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    temp : int or float
        The temperature (in Kelvin) at which the simulation was run. This is used
        to select the corresponding dataset within the H5 file.
    replica : int
        The replica number of the simulation to extract data from. This is used
        to select the corresponding dataset within the H5 file.

    Returns:
    md.Trajectory
        An MDTraj trajectory object containing the loaded protein structure and
        simulation coordinates.

    Example:
    -------
    import h5py
    import mdtraj as md

    # Open the H5 file
    with h5py.File('simulation_data.h5', 'r') as h5file:
        traj = convert_to_mdtraj(h5file, 300, 1)

    # Now 'traj' can be used for analysis with MDTraj
    """    
    h5, code = _open_h5_file(h5)
    pdb_file_name, coords = _extract_structure_and_coordinates(h5, code, temp, replica)
    trj = md.load(pdb_file_name)
    os.unlink(pdb_file_name)
    trj.xyz = coords.copy()
    trj.time = np.arange(1, coords.shape[0] + 1)
    return trj


def convert_to_files(
    h5_path, basename=None, temp_list=[320, 348, 379, 413, 450], replica_list=[0, 1, 2, 3, 4], output_dir=None
):
    """
    Converts data from an H5 file to separate PDB and XTC files based on specified temperatures and replicas.

    This function reads protein atom structures and simulation data from an H5 file and writes a single PDB file
    and multiple XTC files. Each XTC file corresponds to a specific temperature and replica combination. The
    function uses `convert_to_mdtraj` to generate MDTraj trajectory objects which are then saved in the XTC format.

    Parameters:
    h5_path : str
        The file name or path to the H5 file containing the simulation data.
    basename : str
        The base name to use for output files.  If None, it is taken from the domain ID.
    temp_list : list of int, optional
        A list of temperatures (in Kelvin) for which the simulations were run. Defaults to [320, 348, 379, 413, 450].
    replica_list : list of int, optional
        A list of replica numbers to extract data for. Defaults to [0, 1, 2, 3, 4].
    output_dir : str, optional
        Directory to save output files. If None, uses current directory.

    Outputs:
    Creates a PDB file named `{basename}.pdb` and multiple XTC files named `{basename}_{temp}_{replica}.xtc`,
    where `{temp}` and `{replica}` are values from `temp_list` and `replica_list`.

    Example:
    -------
    # Convert data to files with base name 'protein_simulation'
    convert_to_files('simulation_data.h5', 'protein_simulation')
    """
    h5, code = _open_h5_file(h5_path)

    if not basename:
        basename = code

    # Set output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pdbpath = os.path.join(output_dir, f"{basename}.pdb")
    else:
        pdbpath = f"{basename}.pdb"

    # Write PDB file
    pdb_file_name = _extract_structure(h5, code)
    with open(pdbpath, "wb") as pdbfile:
        with open(pdb_file_name, "rb") as temp_pdb:
            pdbfile.write(temp_pdb.read())
        print(f"Wrote {pdbpath}")

    # Write XTC files for each temperature and replica
    for temp in temp_list:
        for replica in replica_list:
            if output_dir:
                xtcpath = os.path.join(output_dir, f"{basename}_{temp}_{replica}.xtc")
            else:
                xtcpath = f"{basename}_{temp}_{replica}.xtc"
            
            trj = convert_to_mdtraj(h5, temp, replica)
            trj.save_xtc(xtcpath)
            print(f"Wrote {xtcpath}")


# ============================================================================
# Download and conversion workflow
# ============================================================================

def download_entry(entry_name, output_dir="./mdcath_data"):
    """
    Download a specific entry from the mdCATH dataset.
    
    Args:
        entry_name: Name of the entry (e.g., "4i69A00_320")
        output_dir: Directory to save downloaded files
    
    Returns:
        Path to the downloaded file, or None if download failed
    """
    # Extract the domain name (remove the temperature suffix _320)
    domain_name = entry_name.rsplit('_', 1)[0]
    
    # Construct the filename - format is data/mdcath_dataset_{domain}.h5
    filename = f"data/mdcath_dataset_{domain_name}.h5"
    
    try:
        # Download the file from HuggingFace Hub
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        return local_path
    except Exception as e:
        print(f"Error downloading {entry_name}: {str(e)}")
        return None


def process_entry(entry_name, h5_dir, convert_dir, temp_list, replica_list):
    """
    Download and optionally convert a single entry.
    
    Args:
        entry_name: Name of the entry (e.g., "4i69A00_320")
        h5_dir: Directory containing H5 files
        convert_dir: Directory for converted files
        temp_list: List of temperatures to convert
        replica_list: List of replicas to convert
        convert: Whether to convert to XTC/PDB
    
    Returns:
        True if successful, False otherwise
    """
    domain_name = entry_name.rsplit('_', 1)[0]
    h5_filename = f"mdcath_dataset_{domain_name}.h5"
    h5_path = os.path.join(h5_dir, "data", h5_filename)
    
    # Check if file exists
    if not os.path.exists(h5_path):
        print(f"Downloading {entry_name}...")
        downloaded_path = download_entry(entry_name, h5_dir)
        if not downloaded_path:
            return False
        h5_path = downloaded_path
    else:
        print(f"H5 file already exists: {h5_filename}")
    
    # Convert if requested
    print(f"Converting {domain_name} to XTC/PDB...")
    convert_to_files(
        h5_path,
        basename=domain_name,
        temp_list=temp_list,
        replica_list=replica_list,
        output_dir=convert_dir
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert specific mdCATH dataset entries from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mdcath/h5",
        help="Directory to save downloaded H5 files (default: ./mdcath_data)"
    )
    parser.add_argument(
        "--convert-dir",
        type=str,
        default="data/mdcath/converted",
        help="Directory to save converted XTC/PDB files (default: ./mdcath_converted)"
    )
    parser.add_argument(
        "--entries",
        type=str,
        nargs="+",
        help="Specific entries to download (default: all predefined entries)"
    )
    parser.add_argument(
        "--temp-list",
        type=int,
        nargs="+",
        default=[320],
        help="List of temperatures to convert (default: [320])"
    )
    parser.add_argument(
        "--replica-list",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of replicas to convert (default: [0, 1, 2, 3, 4])"
    )
    
    args = parser.parse_args()
    
    # Use custom entries if provided, otherwise use the predefined list
    entries_to_process = args.entries if args.entries else ENTRIES
    entries_to_process = sorted(set(entries_to_process))
    entries_to_process = ['1zxfA00']  # For testing purposes, process only one entry

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.convert_dir, exist_ok=True)
    
    print(f"Processing {len(entries_to_process)} entries from mdCATH dataset")
    print(f"H5 files directory: {args.output_dir}")
    print(f"Converted files directory: {args.convert_dir}")
    print(f"Temperatures: {args.temp_list}")
    print(f"Replicas: {args.replica_list}")
    print("-" * 70)
    
    successful = 0
    failed = 0
    
    # Process each entry
    for entry in tqdm(entries_to_process, desc="Processing entries"):
        print(f"\n[{entry}]")
        result = process_entry(
            entry,
            args.output_dir,
            args.convert_dir,
            args.temp_list,
            args.replica_list,
        )
        if result:
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Processing complete!")
    print(f"Successful: {successful}/{len(entries_to_process)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(entries_to_process)}")
    print(f"\nH5 files saved to: {os.path.abspath(args.output_dir)}")
    print(f"Converted files saved to: {os.path.abspath(args.convert_dir)}")


if __name__ == "__main__":
    main()