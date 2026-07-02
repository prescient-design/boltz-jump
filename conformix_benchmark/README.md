# Benchmarking Boltz-Jump on ConforMix Datasets

This directory contains everything needed to benchmark boltz-jump (walk-jump
sampling) against the ConforMix paper's benchmark sets.

## Datasets (from ConforMix repo)

| Dataset               | # Proteins | What it tests                          |
|-----------------------|------------|----------------------------------------|
| `domainmotion`        | 38         | Open/closed domain hinge motions       |
| `membranetransporters`| 15         | Inward-open ↔ outward-open cycling     |
| `crypticpockets`      | 34         | Pocket opening for ligand binding      |
| `foldswitching`       | 15         | Wholesale secondary structure changes  |

Each dataset CSV contains the system ID, two reference PDB structures (the
conformational endpoints), and the RMSD between them.

## Quick Start

### 1. Generate YAML configs

```bash
python scripts/generate_configs.py \
    --datasets_dir . \
    --configs_dir configs
```

This reads the `*_sequences.csv` files and writes one boltz-jump YAML per
protein into `configs/{dataset}/{system_id}.yaml`.

### 2. Download reference PDB structures

```bash
python scripts/download_refs.py \
    --datasets_dir . \
    --ref_pdbs_dir ref_pdbs
```

Downloads all reference CIF files from RCSB (~170 unique PDB codes).

### 3. Run boltz-jump predictions

**Single protein (test):**
```bash
boltz-jump predict configs/domainmotion/P0DP23.yaml \
    --walk_jump_sigma 0.1 \
    --no_kernels \
    --use_msa_server \
    --out_dir results/domainmotion
```

**All proteins via SLURM:**
Edit the paths in `scripts/run_boltz_jump.sh`, then:
```bash
sbatch scripts/run_boltz_jump.sh
```

### 4. Evaluate conformational coverage

```bash
pip install mdtraj pandas  # if not already installed

python scripts/eval_coverage.py \
    --dataset domainmotion \
    --datasets_dir . \
    --predictions_dir results/domainmotion \
    --ref_pdbs_dir ref_pdbs \
    --output_dir eval_results
```

Repeat for each dataset. This computes the key metric from ConforMix Table 1:
**coverage at 50%** — the fraction of proteins where your best sample is within
50% of the RMSD between the two reference structures.

## Key Metric: Coverage

The ConforMix paper reports coverage as follows:

- For each protein, compute min-RMSD from any generated sample to each
  reference structure.
- **Best-matched** = the reference that is easier to match (lower min-RMSD).
- **Worst-matched** = the alternate state (harder to reach).
- **Coverage at X%** = fraction of proteins where min-RMSD ≤ X% × ref-to-ref RMSD.

ConforMix Table 1 results at 50% threshold:

| Method              | Domain motion | Transporters | Cryptic pockets | Fold switch |
|---------------------|---------------|--------------|-----------------|-------------|
| Default Boltz       | 0.33          | 0.13         | 0.15            | 0.13        |
| AFCluster-Boltz     | 0.46          | 0.19         | 0.35            | 0.27        |
| CF-random-Boltz     | 0.51          | 0.20         | 0.39            | 0.20        |
| ConforMixRMSD-Boltz | **0.69**      | **0.33**     | **0.45**        | 0.13        |

(These are worst-matched coverage values — the harder task.)

## Output Structure

```
eval_results/
├── domainmotion_results.csv      # Per-protein RMSD results
├── domainmotion_summary.json     # Coverage curves
├── membranetransporters_results.csv
├── ...
```

## Notes

- The ConforMix RMSD metric uses only Cα atoms in secondary structure
  elements (helices/sheets). The eval script here uses all Cα atoms for
  simplicity. For a stricter comparison, you'd want to add secondary
  structure masking (via DSSP).
- The ConforMix paper also filters samples by pLDDT quality. Your
  boltz-jump pipeline may handle this differently.
- For TM-score computation, install `TMalign` and see ConforMix
  supplementary Figure S4.
