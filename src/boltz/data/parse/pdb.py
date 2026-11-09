from typing import Optional
from rdkit.Chem.rdchem import Mol
import gemmi
from tempfile import NamedTemporaryFile
from boltz.data.parse.mmcif import parse_mmcif, ParsedStructure


def get_chain_sequences(structure: gemmi.Structure) -> dict[str, str]:
    """Extract sequences for each chain in the structure."""
    chain_sequences = {}
    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                if residue.name in {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                                    'THR', 'TRP', 'TYR', 'VAL'}:
                    sequence.append(residue.name)
                elif residue.name in {'HSD', 'HSE', 'HSP'}:
                    sequence.append('HIS')  # Standardize histidine variants
                else:
                    sequence.append('UNK')  # Unknown or non-standard residue
            
            assert chain.name not in chain_sequences, f"Duplicate chain name found: {chain.name}"            
            chain_sequences[chain.name] = '-'.join(sequence)
            print(f"Chain {chain.name}: Sequence {(sequence)}")
    return chain_sequences


def add_entities_to_structure(structure: gemmi.Structure) -> None:
    """Add entities to a Gemmi structure based on unique chain sequences."""
    
    if len(structure.entities) > 0:
        print("Structure already has entities defined. Skipping entity creation.")
        return

    # Get number of unique chains and their sequences
    chain_sequences = get_chain_sequences(structure)
    unique_sequences = set(chain_sequences.values())

    # For each unique sequence, create an entity
    for index, sequence in enumerate(unique_sequences, start=1):
        # Step 2: Manually create a protein entity
        entity = gemmi.Entity(str(index))
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        
        entity.full_sequence = sequence.split('-')
        entity.subchains = [chain_name for chain_name, seq in chain_sequences.items() if seq == sequence]
        structure.entities.append(entity)
    
        print(f"Created entity with sequence: {entity.full_sequence}")
    
        for model in structure:
            for chain in model:
                if chain_sequences[chain.name] == sequence:
                    for residue in chain:
                        residue.entity_id = str(index)
                        residue.subchain = chain.name
        
    print("Assigned entity IDs to all residues")


def parse_pdb(
    path: str,
    mols: Optional[dict[str, Mol]] = None,
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    compute_interfaces: bool = True,
) -> ParsedStructure:
    with NamedTemporaryFile(suffix=".cif") as tmp_cif_file:
        tmp_cif_path = tmp_cif_file.name
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()

        # Delete existing entities and re-create them
        structure.entities.clear()
        add_entities_to_structure(structure)

        subchain_counts, subchain_renaming = {}, {}
        for chain in structure[0]:
            subchain_counts[chain.name] = 0
            for res in chain:
                if res.subchain not in subchain_renaming:
                    subchain_renaming[res.subchain] = chain.name + str(subchain_counts[chain.name] + 1)
                    subchain_counts[chain.name] += 1
                res.subchain = subchain_renaming[res.subchain]
        for entity in structure.entities:
            entity.subchains = [subchain_renaming[subchain] for subchain in entity.subchains]
    
        doc = structure.make_mmcif_document()
        doc.write_file(tmp_cif_path)

        return parse_mmcif(
            path=tmp_cif_path,
            mols=mols,
            moldir=moldir,
            use_assembly=use_assembly,
            compute_interfaces=compute_interfaces
        )