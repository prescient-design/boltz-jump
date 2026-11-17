#!/usr/bin/env python3
"""Prepare the PoseBusters co-folding screen.

For every entry of the PoseBusters benchmark set (Zenodo 8278563,
posebusters_benchmark_set/<PDB>_<LIG>/), extract the protein chain sequences
from <entry>_protein.pdb and the crystal ligand as SMILES from
<entry>_ligand.sdf, then write:

    $DOCKING_DIR/posebusters/manifest.json         per-entry chains, SMILES, heavy atoms
    $DOCKING_DIR/posebusters/sequences.fasta       unique protein sequences (header = hash)
    $DOCKING_DIR/posebusters/cofold_configs/*.yaml Boltz-2 co-folding inputs with MSAs

MSAs are expected at ${DOCKING_DIR}/msa/<hash>.a3m (see
scripts/slurm/docking_msa.sh). Entries without a parseable ligand or without a
protein chain are listed in manifest.json under "skipped".

Usage:
    python scripts/docking/prepare_posebusters.py
"""

import argparse
import json
from pathlib import Path

import gemmi
from rdkit import Chem

from common import docking_dir, msa_path, seq_hash

LIGAND_IDS = list("XYZ0123456789")


def protein_chains(pdb_path):
    st = gemmi.read_structure(str(pdb_path))
    st.setup_entities()
    chains = {}
    for chain in st[0]:
        poly = chain.get_polymer()
        if not len(poly) or poly.check_polymer_type() not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            continue
        seq = gemmi.one_letter_code(poly.extract_sequence()).replace("X", "")
        if len(seq) >= 10:
            chains[chain.name] = seq
    return chains


def ligand_smiles(sdf_path):
    for sanitize in (True, False):
        mols = [m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=sanitize) if m is not None]
        if mols:
            mol = mols[0]
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            return Chem.MolToSmiles(mol), mol.GetNumHeavyAtoms()
    return None, 0


def cofold_yaml(entry):
    lines = ["version: 1", "sequences:"]
    for chain_id, seq in entry["chains"].items():
        lines += ["  - protein:", f"      id: '{chain_id}'", f"      sequence: {seq}",
                  f"      msa: {msa_path(seq)}"]
    ligand_id = next(x for x in LIGAND_IDS if x not in entry["chains"])
    lines += ["  - ligand:", f"      id: '{ligand_id}'", f"      smiles: '{entry['ligand_smiles']}'"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_dir", help="Default: $DOCKING_DIR/data/posebusters/posebusters_benchmark_set")
    args = parser.parse_args()

    root = docking_dir()
    bench = Path(args.benchmark_dir or root / "data/posebusters/posebusters_benchmark_set")
    out = root / "posebusters"
    (out / "cofold_configs").mkdir(parents=True, exist_ok=True)

    entries, skipped = {}, {}
    for edir in sorted(p for p in bench.iterdir() if p.is_dir()):
        name = edir.name
        chains = protein_chains(edir / f"{name}_protein.pdb")
        smiles, n_heavy = ligand_smiles(edir / f"{name}_ligand.sdf")
        if not chains:
            skipped[name] = "no protein chain"
            continue
        if smiles is None:
            skipped[name] = "ligand SMILES parse failure"
            continue
        entries[name] = {"chains": chains, "ligand_smiles": smiles, "ligand_heavy": n_heavy,
                         "protein_pdb": str(edir / f"{name}_protein.pdb"),
                         "ligand_sdf": str(edir / f"{name}_ligand.sdf")}

    seqs = {seq_hash(s): s for e in entries.values() for s in e["chains"].values()}
    (out / "sequences.fasta").write_text("".join(f">{h}\n{s}\n" for h, s in sorted(seqs.items())))
    for name, entry in entries.items():
        (out / "cofold_configs" / f"{name}.yaml").write_text(cofold_yaml(entry))
    json.dump({"entries": entries, "skipped": skipped}, open(out / "manifest.json", "w"), indent=1)
    print(f"{len(entries)} entries ({len(skipped)} skipped: {skipped}); {len(seqs)} unique sequences")


if __name__ == "__main__":
    main()
