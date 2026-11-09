#!/usr/bin/env python3
"""Download ATLAS reference MD trajectories from the official ATLAS server.

Each target is unpacked to ``<out_dir>/<name>/protein/``, which is the layout
expected by ``scripts/eval/analyse_ensembles.py --dataset atlas_test``.
Targets that are already present are skipped.

Usage:
    python scripts/data/download_atlas.py --names atlas_test_names.txt --out_dir data/atlas
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{name}/{name}_protein.zip"
REQUIRED = ["{name}.pdb"] + [f"{{name}}_prod_R{r}_fit.xtc" for r in (1, 2, 3)]


def is_complete(protein_dir: Path, name: str) -> bool:
    return all((protein_dir / f.format(name=name)).exists() for f in REQUIRED)


def download(name: str, out_dir: Path) -> None:
    protein_dir = out_dir / name / "protein"
    if is_complete(protein_dir, name):
        print(f"[skip] {name}")
        return

    protein_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / name / f"{name}_protein.zip"
    print(f"[get]  {name}")
    with urllib.request.urlopen(URL.format(name=name), timeout=120) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)

    with zipfile.ZipFile(zip_path) as zf:
        # Flatten: the archive may or may not contain a top-level directory.
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = protein_dir / Path(member.filename).name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink()

    if not is_complete(protein_dir, name):
        missing = [f.format(name=name) for f in REQUIRED if not (protein_dir / f.format(name=name)).exists()]
        sys.exit(f"{name}: archive is missing {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", type=Path, required=True, help="Text file with one ATLAS name (e.g. 1d3y_B) per line")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="Download only these names")
    args = parser.parse_args()

    names = [line.strip() for line in args.names.read_text().splitlines() if line.strip()]
    if args.only:
        names = [n for n in names if n in args.only]
    for name in names:
        download(name, args.out_dir)


if __name__ == "__main__":
    main()
