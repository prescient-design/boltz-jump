#!/usr/bin/env python3
"""Check that every file referenced by the input configs exists.

Expands ${VAR} references the same way boltz-jump does (from the environment
and .env) and reports MSA / structure paths that are missing.

Usage:
    python scripts/data/check_configs.py configs/atlas_test_msa configs/mdcath_msa ...
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

from boltz.data.parse.yaml import expand_env_vars


def referenced_paths(data):
    for entry in data.get("sequences", []):
        for chain in entry.values():
            msa = chain.get("msa")
            if msa and msa != "empty":
                yield msa
    for key in ("initial_structure", "final_structure"):
        for path in (data.get(key) or {}).values():
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True), override=False)

    n_files, problems = 0, []
    for config_dir in args.config_dirs:
        for path in sorted(config_dir.rglob("*.yaml")):
            n_files += 1
            try:
                data = expand_env_vars(yaml.safe_load(path.read_text()), source=path)
            except ValueError as e:
                problems.append(str(e))
                continue
            problems += [f"{path}: missing {p}" for p in referenced_paths(data) if not Path(p).exists()]

    for p in problems:
        print(p)
    print(f"Checked {n_files} configs: {len(problems)} problems.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
