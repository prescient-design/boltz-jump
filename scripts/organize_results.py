#!/usr/bin/env python3
"""Organize boltz-jump results into results_organized/ for analyse_ensembles.py.

Skips any target/model combination that already exists in results_organized/.
Run from the repo root:
    python scripts/organize_results.py
"""

import os
import mdtraj as md
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)


OUTPUT_DIR = os.environ.get("RESULTS_DIR", "results")
NEW_OUTPUT_DIR = "results_organized"

# (results_subdir,                  organized_dataset,  model_name,                      prefix)
EXPERIMENTS = [
    # --- Existing ---
    ("atlas_test_msa",               "atlas_test_msa",   "boltz2_full_msa",               "boltz_results_subsample_msa=False"),
    ("mdcath_msa",                   "mdcath_msa",       "boltz2_full_msa",               "boltz_results_subsample_msa=False"),
    ("atlas_test_msa", "atlas_test_msa", "boltz2_msa_subsample_32", "boltz_results_subsample_msa=True_num_subsampled=32"),
    ("mdcath_msa", "mdcath_msa", "boltz2_msa_subsample_32", "boltz_results_subsample_msa=True_num_subsampled=32"),
    ("atlas_test_msa", "atlas_test_msa", "boltz2_msa_subsample_64", "boltz_results_subsample_msa=True_num_subsampled=64"),
    ("mdcath_msa", "mdcath_msa", "boltz2_msa_subsample_64", "boltz_results_subsample_msa=True_num_subsampled=64"),
    ("atlas_test_msa",               "atlas_test_msa",   "boltz2_diffusion",              "boltz_diffusion"),
    ("mdcath_msa",                   "mdcath_msa",       "boltz2_diffusion",              "boltz_diffusion"),
    ("atlas_test_msa",               "atlas_test_msa",   "boltz2_jump",                   "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("mdcath_msa",                   "mdcath_msa",       "boltz2_jump",                   "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    # --- Diffusion empty MSA ---
    ("atlas_test_empty_msa",         "atlas_test_msa",   "boltz2_diffusion_empty_msa",    "boltz_results_subsample_msa=False"),
    ("mdcath_empty_msa",             "mdcath_msa",       "boltz2_diffusion_empty_msa",    "boltz_results_subsample_msa=False"),
    # --- MSA ablation ---
    ("atlas_test_msa",               "atlas_test_msa",   "boltz2_jump_msa",               "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("atlas_test_empty_msa",         "atlas_test_msa",   "boltz2_jump_empty_msa",         "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("atlas_test_msa_no_init",       "atlas_test_msa",   "boltz2_jump_msa_no_init",       "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("atlas_test_empty_msa_no_init", "atlas_test_msa",   "boltz2_jump_empty_msa_no_init", "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("mdcath_msa",                   "mdcath_msa",       "boltz2_jump_msa",               "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("mdcath_empty_msa",             "mdcath_msa",       "boltz2_jump_empty_msa",         "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("mdcath_msa_no_init",           "mdcath_msa",       "boltz2_jump_msa_no_init",       "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("mdcath_empty_msa_no_init",     "mdcath_msa",       "boltz2_jump_empty_msa_no_init", "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    # --- Walk-jump with Boltz-2 init + empty MSA ---
    ("atlas_test_empty_msa_boltz2_init", "atlas_test_msa", "boltz2_jump_empty_msa_boltz2_init", "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    ("mdcath_empty_msa_boltz2_init",     "mdcath_msa",     "boltz2_jump_empty_msa_boltz2_init", "boltz_jump_results_s=2.0_d=2.0_f=0.1"),
    # --- d=2.0 walk-jump runs truncated to 100 frames (for fair comparison with d=1.0) ---
    ("atlas_test_msa",               "atlas_test_msa",   "boltz2_jump_msa_d2_100",               "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("atlas_test_empty_msa",         "atlas_test_msa",   "boltz2_jump_empty_msa_d2_100",         "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("atlas_test_msa_no_init",       "atlas_test_msa",   "boltz2_jump_msa_no_init_d2_100",       "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("atlas_test_empty_msa_no_init", "atlas_test_msa",   "boltz2_jump_empty_msa_no_init_d2_100", "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("atlas_test_empty_msa_boltz2_init", "atlas_test_msa", "boltz2_jump_empty_msa_boltz2_init_d2_100", "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("mdcath_msa",                   "mdcath_msa",       "boltz2_jump_msa_d2_100",               "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("mdcath_empty_msa",             "mdcath_msa",       "boltz2_jump_empty_msa_d2_100",         "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("mdcath_msa_no_init",           "mdcath_msa",       "boltz2_jump_msa_no_init_d2_100",       "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("mdcath_empty_msa_no_init",     "mdcath_msa",       "boltz2_jump_empty_msa_no_init_d2_100", "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    ("mdcath_empty_msa_boltz2_init", "mdcath_msa",       "boltz2_jump_empty_msa_boltz2_init_d2_100", "boltz_jump_results_s=2.0_d=2.0_f=0.1", 100),
    # --- d=1.0 walk-jump runs ---
    ("atlas_test_msa",               "atlas_test_msa",   "boltz2_jump_msa_d1",               "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("atlas_test_empty_msa",         "atlas_test_msa",   "boltz2_jump_empty_msa_d1",         "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("atlas_test_msa_no_init",       "atlas_test_msa",   "boltz2_jump_msa_no_init_d1",       "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("atlas_test_empty_msa_no_init", "atlas_test_msa",   "boltz2_jump_empty_msa_no_init_d1", "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("atlas_test_empty_msa_boltz2_init", "atlas_test_msa", "boltz2_jump_empty_msa_boltz2_init_d1", "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("mdcath_msa",                   "mdcath_msa",       "boltz2_jump_msa_d1",               "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("mdcath_empty_msa",             "mdcath_msa",       "boltz2_jump_empty_msa_d1",         "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("mdcath_msa_no_init",           "mdcath_msa",       "boltz2_jump_msa_no_init_d1",       "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("mdcath_empty_msa_no_init",     "mdcath_msa",       "boltz2_jump_empty_msa_no_init_d1", "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    ("mdcath_empty_msa_boltz2_init", "mdcath_msa",       "boltz2_jump_empty_msa_boltz2_init_d1", "boltz_jump_results_s=2.0_d=1.0_f=0.1"),
    # --- Nov 2026 re-runs (seeded): fixed Boltz-2+MSA seeds, fixed noised-seed units, replicates ---
    ("atlas_test_empty_msa_boltz2_init_v2", "atlas_test_msa", "boltz2_jump_empty_msa_boltz2_init_d1_v2", "boltz_jump_results_s=2.0_d=1.0_f=0.1_"),
    ("mdcath_empty_msa_boltz2_init_v2", "mdcath_msa", "boltz2_jump_empty_msa_boltz2_init_d1_v2", "boltz_jump_results_s=2.0_d=1.0_f=0.1_"),
    ("noised_seed_sde_atlas_v2", "atlas_test_msa", "boltz2_noised_seed_sde_v2", "boltz_noised_seed_sde_s=2.0"),
    ("noised_seed_ode_atlas_v2", "atlas_test_msa", "boltz2_noised_seed_ode_v2", "boltz_noised_seed_ode_s=2.0"),
    ("noised_seed_sde_mdcath_v2", "mdcath_msa", "boltz2_noised_seed_sde_v2", "boltz_noised_seed_sde_s=2.0"),
    ("noised_seed_ode_mdcath_v2", "mdcath_msa", "boltz2_noised_seed_ode_v2", "boltz_noised_seed_ode_s=2.0"),
    ("atlas_test_msa_rep1", "atlas_test_msa", "boltz2_jump_msa_d1_rep1", "boltz_jump_results_s=2.0_d=1.0_f=0.1_"),
    ("atlas_test_msa_rep2", "atlas_test_msa", "boltz2_jump_msa_d1_rep2", "boltz_jump_results_s=2.0_d=1.0_f=0.1_"),
    ("mdcath_msa_rep1", "mdcath_msa", "boltz2_jump_msa_d1_rep1", "boltz_jump_results_s=2.0_d=1.0_f=0.1_"),
    ("mdcath_msa_rep2", "mdcath_msa", "boltz2_jump_msa_d1_rep2", "boltz_jump_results_s=2.0_d=1.0_f=0.1_"),
    ("atlas_test_msa", "atlas_test_msa", "boltz2_walk_denoise_n10", "boltz_jump_results_s=2.0_d=1.0_f=0.1_n=10_"),
    ("atlas_test_msa", "atlas_test_msa", "boltz2_walk_denoise_n53", "boltz_jump_results_s=2.0_d=1.0_f=0.1_n=53_"),
    ("mdcath_msa", "mdcath_msa", "boltz2_walk_denoise_n10", "boltz_jump_results_s=2.0_d=1.0_f=0.1_n=10_"),
    ("mdcath_msa", "mdcath_msa", "boltz2_walk_denoise_n53", "boltz_jump_results_s=2.0_d=1.0_f=0.1_n=53_"),
    ("noised_seed_sde_atlas",        "atlas_test_msa",   "boltz2_noised_seed_sde",        "boltz_noised_seed_sde_s=2.0"),
    ("noised_seed_ode_atlas",        "atlas_test_msa",   "boltz2_noised_seed_ode",        "boltz_noised_seed_ode_s=2.0"),
    ("noised_seed_one_step_atlas",   "atlas_test_msa",   "boltz2_noised_seed_one_step",   "boltz_noised_seed_one_step_s=2.0"),
    ("noised_seed_sde_mdcath",       "mdcath_msa",       "boltz2_noised_seed_sde",        "boltz_noised_seed_sde_s=2.0"),
    ("noised_seed_ode_mdcath",       "mdcath_msa",       "boltz2_noised_seed_ode",        "boltz_noised_seed_ode_s=2.0"),
    ("noised_seed_one_step_mdcath",  "mdcath_msa",       "boltz2_noised_seed_one_step",   "boltz_noised_seed_one_step_s=2.0"),
]


def process_experiment(results_subdir, organized_dataset, model_name, prefix, max_frames=None):
    dataset_dir = os.path.join(OUTPUT_DIR, results_subdir)
    if not os.path.isdir(dataset_dir):
        print(f"  [skip] {results_subdir}: directory not found")
        return

    for result_dir in sorted(os.listdir(dataset_dir)):
        if not result_dir.startswith(prefix):
            continue
        # Walk-denoise runs ("..._f=0.1_n=10_<target>") share the single-step
        # prefix; only match them when the prefix asks for them explicitly.
        # (Noised-seed directories also contain "_n=<samples>_" and are not affected.)
        if (prefix.startswith("boltz_jump_results") and "n=" not in prefix
                and result_dir[len(prefix):].lstrip("_").startswith("n=")):
            continue

        predictions_dir = os.path.join(dataset_dir, result_dir, "predictions")
        if not os.path.isdir(predictions_dir) or not os.listdir(predictions_dir):
            continue

        systems = os.listdir(predictions_dir)
        assert len(systems) == 1, f"Expected 1 system in {predictions_dir}, got {systems}"
        system = systems[0]

        out_dir = Path(NEW_OUTPUT_DIR) / organized_dataset / system / model_name

        if out_dir.exists():
            print(f"  [skip] {organized_dataset}/{system}/{model_name}: already exists")
            continue

        cif_file = os.path.join(predictions_dir, system, f"{system}_model_0.cif")
        xtc_file = os.path.join(predictions_dir, system, f"{system}_model_0.xtc")

        if not os.path.isfile(cif_file):
            print(f"  [skip] {organized_dataset}/{system}/{model_name}: missing cif")
            continue

        if os.path.isfile(xtc_file):
            traj = md.load(xtc_file, top=cif_file)
        else:
            # Multi-CIF diffusion results: load and concatenate all model CIFs
            import glob as _glob
            cif_files = sorted(
                _glob.glob(os.path.join(predictions_dir, system, f"{system}_model_*.cif")),
                key=lambda p: int(p.split("_model_")[-1].replace(".cif", ""))
            )
            if not cif_files:
                print(f"  [skip] {organized_dataset}/{system}/{model_name}: no cif files found")
                continue
            frames = [md.load(f) for f in cif_files]
            traj = md.join(frames)
        if max_frames is not None:
            traj = traj[:max_frames]
        out_dir.mkdir(parents=True, exist_ok=True)
        traj.save_xtc(str(out_dir / "traj.xtc"))
        traj[0].save_cif(str(out_dir / "system.cif"))
        print(f"  [wrote] {organized_dataset}/{system}/{model_name}: {traj}")


ALPHAFLOW_DIR = os.environ.get("ALPHAFLOW_OUTPUT_DIR", "")
ALPHAFLOW_DATASETS = {
    "atlas_test": "atlas_test_msa",
    "mdcath":     "mdcath_msa",
}

BIOEMU_DIR = os.environ.get("BIOEMU_OUTPUT_DIR", "")
BIOEMU_DATASETS = {
    "atlas":   "atlas_test_msa",
    "mdcath":  "mdcath_msa",
}


def process_alphaflow():
    for dataset, organized_dataset in ALPHAFLOW_DATASETS.items():
        dataset_dir = os.path.join(ALPHAFLOW_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"  [skip] alphaflow {dataset}: directory not found")
            continue
        for model_type in ["md_distilled", "md_base"]:
            model_name = f"alphaflow_{model_type}"
            model_dir = os.path.join(dataset_dir, model_type)
            if not os.path.isdir(model_dir):
                print(f"  [skip] alphaflow/{dataset}/{model_type}: directory not found")
                continue
            for system_pdb in sorted(os.listdir(model_dir)):
                system = system_pdb.replace(".pdb", "")
                out_dir = Path(NEW_OUTPUT_DIR) / organized_dataset / system / model_name
                if out_dir.exists():
                    print(f"  [skip] {organized_dataset}/{system}/{model_name}: already exists")
                    continue
                traj = md.load(os.path.join(model_dir, system_pdb))
                out_dir.mkdir(parents=True, exist_ok=True)
                traj.save_xtc(str(out_dir / "traj.xtc"))
                traj[0].save_cif(str(out_dir / "system.cif"))
                print(f"  [wrote] {organized_dataset}/{system}/{model_name}: {traj}")


def process_bioemu():
    for dataset, organized_dataset in BIOEMU_DATASETS.items():
        dataset_dir = os.path.join(BIOEMU_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"  [skip] bioemu {dataset}: directory not found")
            continue
        for system in sorted(os.listdir(dataset_dir)):
            system_name = system.replace("_320", "") if dataset == "mdcath" else system
            model_name = "bioemu"
            out_dir = Path(NEW_OUTPUT_DIR) / organized_dataset / system_name / model_name
            if out_dir.exists():
                print(f"  [skip] {organized_dataset}/{system_name}/{model_name}: already exists")
                continue
            system_dir = os.path.join(dataset_dir, system)
            xtc_file = os.path.join(system_dir, "samples_sidechain_rec.xtc")
            pdb_file = os.path.join(system_dir, "samples_sidechain_rec.pdb")
            if not os.path.isfile(xtc_file) or not os.path.isfile(pdb_file):
                print(f"  [skip] {organized_dataset}/{system_name}/{model_name}: missing files")
                continue
            traj = md.load(xtc_file, top=pdb_file)
            out_dir.mkdir(parents=True, exist_ok=True)
            traj.save_xtc(str(out_dir / "traj.xtc"))
            traj[0].save_cif(str(out_dir / "system.cif"))
            print(f"  [wrote] {organized_dataset}/{system_name}/{model_name}: {traj}")


if __name__ == "__main__":
    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)

    print("=== Boltz experiments ===")
    for experiment in EXPERIMENTS:
        process_experiment(*experiment[:4], max_frames=experiment[4] if len(experiment) > 4 else None)

    print("\n=== AlphaFlow ===")
    process_alphaflow()

    print("\n=== BioEMU ===")
    process_bioemu()
