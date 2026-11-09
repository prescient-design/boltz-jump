import os
import re
from pathlib import Path

import yaml
from rdkit.Chem.rdchem import Mol

from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target


_ENV_VAR = re.compile(r"\$\{(\w+)\}")


def expand_env_vars(obj, source: Path = None):
    """Recursively substitute ``${VAR}`` in string values from the environment.

    Configs reference machine-specific locations (datasets, MSAs, results) this
    way so that they can be shared; values are read from ``.env`` by the CLI.
    """
    if isinstance(obj, str):
        missing = sorted({v for v in _ENV_VAR.findall(obj) if v not in os.environ})
        if missing:
            msg = (
                f"Environment variable(s) {', '.join(missing)} used in {source} are not set. "
                "Define them in .env (see .env.example)."
            )
            raise ValueError(msg)
        return _ENV_VAR.sub(lambda m: os.environ[m.group(1)], obj)
    if isinstance(obj, dict):
        return {k: expand_env_vars(v, source) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v, source) for v in obj]
    return obj


def parse_yaml(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    templates:
        - path: /path/to/template.pdb
          ids: [A] # optional, specify which chains to template

    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)
    data = expand_env_vars(data, source=path)

    name = path.stem
    return parse_boltz_schema(name, data, ccd, mol_dir, boltz2)
