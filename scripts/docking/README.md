# Ligand pose refinement

Walk-jump refinement of mis-docked protein–ligand poses, on two benchmarks:

- **PoseBusters** ([Zenodo 8278563](https://zenodo.org/records/8278563), CC BY 4.0)
- **Runs N' Poses** ([Zenodo 18366081](https://zenodo.org/records/18366081))

The task is pose *refinement*, not docking: we keep only systems where Boltz-2
co-folding puts the protein and the binding site right but the ligand pose
wrong, then seed walk-jump from that pose and ask whether it moves the ligand
towards the crystal pose.

Set `DOCKING_DIR` (in `.env`) to a directory for the data and outputs.

## Defining the targets

A system is a **pose failure** when, over all co-folded samples, the best
ligand RMSD is > 2 Å while that same sample has pocket C-alpha RMSD <= 2 Å and
pocket Jaccard >= 0.3 (right fold, right pocket, wrong pose).

For Runs N' Poses the dataset authors publish their own Boltz-2 predictions and
OpenStructure scores, so the failure set is read directly from those — they are
MSA-conditioned, so no screening compute is needed:

```bash
python scripts/docking/prepare_runs_n_poses.py        # -> targets.csv, starts.csv, configs/
```

For PoseBusters the co-folding must be run here:

```bash
python scripts/docking/prepare_posebusters.py         # -> manifest.json, sequences.fasta, cofold_configs/
FASTA=$DOCKING_DIR/posebusters/sequences.fasta OUT_DIR=$DOCKING_DIR/msa \
    scripts/slurm/submit.sh gpu scripts/slurm/docking_msa.sh
scripts/slurm/submit.sh gpu scripts/slurm/docking_cofold.sh --array=0-427
python scripts/docking/score_cofold.py                # -> screen.csv, failures.csv
```

**No post-hoc filtering.** Every system meeting the failure definition is kept.
Do not drop targets by how far the seed sits from the crystal pose: seed
distance predicts success, so filtering on it inflates the reported rate.

**Screen with the same MSA setting you report.** Screening without MSAs is much
cheaper, but it selects systems that are hard for a weaker model than the one
being evaluated, which biases the comparison.

## MSAs

Looked up by sequence hash. By default `$DOCKING_DIR/msa/<hash>.a3m`, as written
by `scripts/slurm/docking_msa.sh` (GPU `mmseqs` against the ColabFold databases;
needs `COLABFOLD_DB_DIR`). To reuse an existing collection set `MSA_DIR`, plus
`MSA_LAYOUT=csv` if it stores `<hash>/A.csv` instead.

## Sampling and scoring

```bash
python scripts/docking/prepare_walkjump.py runs_n_poses --sigmas 2 4 6 8 --seeds 1
TASKS=$DOCKING_DIR/runs_n_poses/walkjump_tasks.tsv \
    scripts/slurm/submit.sh gpu scripts/slurm/docking_walkjump.sh --array=0-<n-1>
python scripts/docking/analyze_walkjump.py runs_n_poses   # -> analysis/{frames,targets,sigmas}.csv
```

Each run is 200 frames at one sigma from one start, with step size `0.2 * sigma`
and friction 2.0. `sigmas.csv` reports targets rescued per sigma, which is how a
single sigma is chosen for all ligand sizes rather than one per size class.

A frame is **eligible** if the ligand has not collapsed (diameter > 0.7x the
diameter of an ideal conformer built from its SMILES) and no ligand heavy atom
is within 1.5 Å of another heavy atom. Eligible frames are **idealized** by
fitting a library of 100 ETKDG/MMFF conformers onto the sampled coordinates,
which repairs the covalent geometry distorted by the one-step jump; the
reported RMSD is of that idealized pose. Ligand RMSD is symmetry-aware and
computed after superposing the predicted protein on the crystal pocket, trying
every predicted chain so homomultimers are handled.

Note that `ideal_rmsd` is an *oracle*: the best pose in the pool, chosen using
the crystal structure. It measures whether sampling can reach a good pose, not
whether a good pose can be identified without the answer. Selecting the pose
blind is a separate, unsolved problem; `frames.csv` carries the per-frame
confidences needed to evaluate a selector.

## Checks

`tests/test_docking_scoring.py` covers the scorer (symmetry handling, bond
diagnostics, template fit). Separately, this implementation was compared with
the one used for the original campaign by re-scoring its published co-folded
seed poses for Runs N' Poses: 60/60 agreed to within 0.005 Å (median 0.002 Å).
