# https://github.com/bjing2016/alphaflow/blob/master/scripts/analyze_ensembles.py

import argparse
from sklearn.decomposition import PCA
import mdtraj, pickle, tqdm, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import logging

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(levelname)s]: %(message)s"
)


def get_pca(xyz: np.ndarray) -> tuple[PCA, np.ndarray]:
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords


def get_lddt(
    pred_traj: mdtraj.Trajectory, ref_traj: mdtraj.Trajectory, cutoff: int = 15
) -> np.ndarray:
    """Compute lDDT between two trajectories."""
    n_frames1, n_atoms1, _ = pred_traj.xyz.shape
    n_frames2, n_atoms2, _ = ref_traj.xyz.shape
    assert n_atoms1 == n_atoms2, "Number of atoms must be the same: {} vs {}".format(
        n_atoms1, n_atoms2
    )
    n_atoms = n_atoms1

    # Compute distance matrices for all frames
    ref_distmats = (
        np.linalg.norm(
            ref_traj.xyz[:, :, None, :] - ref_traj.xyz[:, None, :, :], axis=-1
        )
        * 10
    )  # (n_frames2, n_atoms, n_atoms)

    pred_distmats = (
        np.linalg.norm(
            pred_traj.xyz[:, :, None, :] - pred_traj.xyz[:, None, :, :], axis=-1
        )
        * 10
    )  # (n_frames1, n_atoms, n_atoms)

    # Create masks for valid neighbors: (n_frames2, n_atoms, n_atoms)
    masks = (ref_distmats > 0.5) & (ref_distmats < cutoff)

    ref_dists_bc = ref_distmats[None, :, :, :]  # (1, n_frames2, n_atoms, n_atoms)
    pred_dists_bc = pred_distmats[:, None, :, :]  # (n_frames1, 1, n_atoms, n_atoms)
    masks_bc = masks[None, :, :, :]  # (1, n_frames2, n_atoms, n_atoms)

    dist_diff = np.abs(pred_dists_bc - ref_dists_bc)
    valid_diff = np.where(masks_bc, dist_diff, np.inf)

    thresholds = [0.5, 1.0, 2.0, 4.0]
    preserved_counts = np.stack(
        [(valid_diff < threshold).sum(axis=-1) for threshold in thresholds], axis=-1
    )  # (n_frames1, n_frames2, n_atoms, 4)

    valid_neighbor_counts = np.maximum(masks_bc.sum(axis=-1), 1)
    fractions = preserved_counts / valid_neighbor_counts[..., None]
    lddt_per_atom = fractions.mean(axis=-1)

    no_neighbors = masks_bc.sum(axis=-1) == 0
    lddt_per_atom = np.where(no_neighbors, 0.0, lddt_per_atom)

    return lddt_per_atom.mean(axis=-1)  # (n_frames1, n_frames2)


def calculate_lddt_precision(
    pred_traj: mdtraj.Trajectory, ref_traj: mdtraj.Trajectory
) -> float:
    lddt_scores = get_lddt(pred_traj, ref_traj)
    precision_scores = lddt_scores.max(axis=1)
    precision = precision_scores.mean()
    return precision


def calculate_lddt_recall(
    pred_traj: mdtraj.Trajectory, ref_traj: mdtraj.Trajectory
) -> float:
    return calculate_lddt_precision(ref_traj, pred_traj)


def calculate_lddt_diversity(pred_traj: mdtraj.Trajectory) -> float:
    lddt_scores = get_lddt(pred_traj, pred_traj)
    diversity_scores = lddt_scores[np.triu_indices(lddt_scores.shape[0], k=1)]
    diversity = (1 - diversity_scores).mean()
    return diversity


def get_rmsds(
    traj1: mdtraj.Trajectory, traj2: mdtraj.Trajectory, broadcast: bool = False
) -> np.ndarray:
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:, None], traj2[None]
    distmat = np.square(traj1 - traj2).sum(-1) ** 0.5 / n_atoms**0.5 * 10
    return distmat


def condense_sidechain_sasas(sasas: np.ndarray, top: mdtraj.Topology) -> np.ndarray:
    assert top.n_residues > 1

    if top.n_atoms != sasas.shape[1]:
        raise ValueError(
            f"The number of atoms in top ({top.n_atoms}) didn't match the "
            f"number of SASAs provided ({sasas.shape[1]}). Make sure you "
            f"computed atom-level SASAs (mode='atom') and that you've passed "
            "the correct topology file and array of SASAs"
        )

    sc_mask = np.array([a.name not in ["CA", "C", "N", "O", "OXT"] for a in top.atoms])
    res_id = np.array([a.residue.index for a in top.atoms])

    rsd_sasas = np.zeros((sasas.shape[0], top.n_residues), dtype="float32")

    for i in range(top.n_residues):
        rsd_sasas[:, i] = sasas[:, sc_mask & (res_id == i)].sum(1)
    return rsd_sasas


def sasa_mi(sasa: np.ndarray) -> np.ndarray:
    N, L = sasa.shape
    joint_probs = np.zeros((L, L, 2, 2))

    joint_probs[:, :, 1, 1] = (sasa[:, :, None] & sasa[:, None, :]).mean(0)
    joint_probs[:, :, 1, 0] = (sasa[:, :, None] & ~sasa[:, None, :]).mean(0)
    joint_probs[:, :, 0, 1] = (~sasa[:, :, None] & sasa[:, None, :]).mean(0)
    joint_probs[:, :, 0, 0] = (~sasa[:, :, None] & ~sasa[:, None, :]).mean(0)

    marginal_probs = np.stack([1 - sasa.mean(0), sasa.mean(0)], -1)
    indep_probs = marginal_probs[None, :, None, :] * marginal_probs[:, None, :, None]
    mi = np.nansum(joint_probs * np.log(joint_probs / indep_probs), (-1, -2))
    mi[np.arange(L), np.arange(L)] = 0
    return mi


def get_mean_covar(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = xyz.mean(0)
    xyz = xyz - mean
    covar = (xyz[..., None] * xyz[..., None, :]).mean(0)
    return mean, covar


def sqrtm(M: np.ndarray) -> np.ndarray:
    D, P = np.linalg.eig(M)
    out = (P * np.sqrt(D[:, None])) @ np.linalg.inv(P)
    return out


def get_wasserstein(distmat: np.ndarray, p: int = 2) -> float:
    assert distmat.shape[0] == distmat.shape[1]
    distmat = distmat**p
    row_ind, col_ind = linear_sum_assignment(distmat)
    return distmat[row_ind, col_ind].mean() ** (1 / p)


def align_tops(
    top1: mdtraj.Topology, top2: mdtraj.Topology
) -> tuple[list[int], list[int]]:
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]

    intersection = [nam for nam in names1 if nam in names2]

    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    return mask1, mask2


def main(args):
    name = args.name
    model = args.model
    logging.info(f"Analyzing {name} with model {model}")
    
    out = {}

    logging.info("Loading reference trajectory.")
    if args.dataset == "atlas_test":
        topfile = f"{args.reference_dir}/{name}/protein/{name}.pdb"
        ref_aa = mdtraj.load(topfile)
        traj_aa = (
            mdtraj.load(
                f"{args.reference_dir}/{name}/protein/{name}_prod_R1_fit.xtc", top=topfile
            )
            + mdtraj.load(
                f"{args.reference_dir}/{name}/protein/{name}_prod_R2_fit.xtc", top=topfile
            )
            + mdtraj.load(
                f"{args.reference_dir}/{name}/protein/{name}_prod_R3_fit.xtc", top=topfile
            )
        )
    elif args.dataset == "mdcath":
        topfile = f"{args.reference_dir}/{name}.pdb"
        ref_aa = mdtraj.load(topfile)
        traj_aa = mdtraj.load(
            f"{args.reference_dir}/{name}_320_0.xtc", top=topfile
        ) + mdtraj.load(
            f"{args.reference_dir}/{name}_320_1.xtc", top=topfile
        ) + mdtraj.load(
            f"{args.reference_dir}/{name}_320_2.xtc", top=topfile
        ) + mdtraj.load(
            f"{args.reference_dir}/{name}_320_3.xtc", top=topfile
        ) + mdtraj.load(
            f"{args.reference_dir}/{name}_320_4.xtc", top=topfile
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logging.info(f"Loaded {traj_aa.n_frames} reference frames")

    logging.info("Loading predicted conformers.")
    print(f"{args.predictions_dir}/{name}/{model}/system.cif")
    predtraj_aa = mdtraj.load(
        f"{args.predictions_dir}/{name}/{model}/traj.xtc",
        top=f"{args.predictions_dir}/{name}/{model}/system.cif",
    )
    logging.info(f"Loaded {predtraj_aa.n_frames} predicted conformers")

    logging.info(f"- Crystal structure has {ref_aa.n_atoms} atoms")
    logging.info(f"- Reference trajectory has {traj_aa.n_atoms} atoms")
    logging.info(f"- Predicted trajectory has {predtraj_aa.n_atoms} atoms")

    # Normalize residue numbering.
    logging.info("Normalizing residue numbering")
    for i, residue in enumerate(traj_aa.top.residues, start=1):  
        residue.resSeq = i
    for i, residue in enumerate(ref_aa.top.residues, start=1):  
        residue.resSeq = i
    for i, residue in enumerate(predtraj_aa.top.residues, start=1):  
        residue.resSeq = i

    logging.info("Removing hydrogens")
    traj_aa.atom_slice(
        [a.index for a in traj_aa.top.atoms if a.element.symbol != "H"], True
    )
    ref_aa.atom_slice(
        [a.index for a in ref_aa.top.atoms if a.element.symbol != "H"], True
    )
    predtraj_aa.atom_slice(
        [a.index for a in predtraj_aa.top.atoms if a.element.symbol != "H"], True
    )

    logging.info(f"- Crystal structure has {ref_aa.n_atoms} atoms")
    logging.info(f"- Reference trajectory has {traj_aa.n_atoms} atoms")
    logging.info(f"- Predicted trajectory has {predtraj_aa.n_atoms} atoms")

    if args.bb_only:
        logging.info("Removing sidechains")
        predtraj_aa.atom_slice(
            [
                a.index
                for a in predtraj_aa.top.atoms
                if a.name in ["CA", "C", "N", "O", "OXT"]
            ],
            True,
        )
        logging.info(f"Predicted trajectory has {predtraj_aa.n_atoms} atoms")

    elif args.ca_only:
        logging.info("Removing all but C-alphas")
        predtraj_aa.atom_slice(
            [a.index for a in predtraj_aa.top.atoms if a.name == "CA"], True
        )
        logging.info(f"Predicted trajectory has {predtraj_aa.n_atoms} atoms")

    refmask, predmask = align_tops(traj_aa.top, predtraj_aa.top)
    if len(predmask) <= 0.8 * predtraj_aa.n_atoms:
        raise ValueError(
            f"Only {len(predmask)} / {predtraj_aa.n_atoms} atoms could be aligned "
            f"between reference and predicted structures."
        )
    traj_aa.atom_slice(refmask, True)
    ref_aa.atom_slice(refmask, True)
    predtraj_aa.atom_slice(predmask, True)

    logging.info(f"Aligned on {predtraj_aa.n_atoms} atoms")

    np.random.seed(137)
    RAND1 = np.random.randint(0, traj_aa.n_frames, predtraj_aa.n_frames)
    RAND2 = np.random.randint(0, traj_aa.n_frames, predtraj_aa.n_frames)
    RAND1K = np.random.randint(0, traj_aa.n_frames, 1000)

    logging.info("Superposing all-atom trajectories to crystal structure")
    traj_aa = traj_aa.superpose(ref_aa)
    predtraj_aa = predtraj_aa.superpose(ref_aa)

    out["ca_mask"] = ca_mask = [a.index for a in traj_aa.top.atoms if a.name == "CA"]
    traj = traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    predtraj = predtraj_aa.atom_slice(ca_mask, False)
    logging.info(f"Sliced {predtraj.n_atoms} C-alphas")

    logging.info("Superposing CA trajectories to crystal structure")
    traj = traj.superpose(ref)
    predtraj = predtraj.superpose(ref)

    logging.info("Computing lDDT metrics")
    # Randomly subsample both pred and ref for lDDT to keep (n_pred, n_ref, n_atoms, n_atoms) tractable.
    max_lddt_frames = args.max_pred_frames if args.max_pred_frames is not None else 100
    pred_lddt_idx = np.random.choice(predtraj.n_frames, min(max_lddt_frames, predtraj.n_frames), replace=False)
    ref_lddt_idx = np.random.choice(len(traj), min(max_lddt_frames, len(traj)), replace=False)
    predtraj_lddt = predtraj[pred_lddt_idx]
    traj_lddt = traj[ref_lddt_idx]

    pred_lddts_to_ref = get_lddt(predtraj_lddt, ref)
    out["mean_pred_lddt_to_ref"] = pred_lddts_to_ref.mean(0)
    out["var_pred_lddt_to_ref"] = pred_lddts_to_ref.var(0)

    traj_lddts_to_ref = get_lddt(traj_lddt, ref)
    out["mean_ref_lddt_to_ref"] = traj_lddts_to_ref.mean(0)
    out["var_ref_lddt_to_ref"] = traj_lddts_to_ref.var(0)

    out["pred_lddt_precision"] = calculate_lddt_precision(predtraj_lddt, traj_lddt)
    out["pred_lddt_recall"] = calculate_lddt_recall(predtraj_lddt, traj_lddt)
    out["pred_lddt_diversity"] = calculate_lddt_diversity(predtraj_lddt)
    out["ref_lddt_diversity"] = calculate_lddt_diversity(traj_lddt)

    logging.info(f"Doing PCA")
    n_atoms = predtraj.n_atoms
    ref_pca, ref_coords = get_pca(traj.xyz)
    pred_coords_ref_pca = ref_pca.transform(predtraj.xyz.reshape(predtraj.n_frames, -1))
    seed_coords_ref_pca = ref_pca.transform(ref.xyz.reshape(1, -1))

    pred_pca, pred_coords = get_pca(predtraj.xyz)
    ref_coords_pred_pca = pred_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_pred_pca = pred_pca.transform(ref.xyz.reshape(1, -1))

    joint_pca, _ = get_pca(np.concatenate([traj[RAND1].xyz, predtraj.xyz]))
    pred_coords_joint_pca = joint_pca.transform(
        predtraj.xyz.reshape(predtraj.n_frames, -1)
    )
    ref_coords_joint_pca = joint_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_joint_pca = joint_pca.transform(ref.xyz.reshape(1, -1))

    out["ref_variance"] = ref_pca.explained_variance_ / n_atoms * 100
    out["pred_variance"] = pred_pca.explained_variance_ / n_atoms * 100
    out["joint_variance"] = joint_pca.explained_variance_ / n_atoms * 100

    out["pred_rmsf"] = mdtraj.rmsf(predtraj_aa, ref_aa) * 10
    out["ref_rmsf"] = mdtraj.rmsf(traj_aa, ref_aa) * 10

    logging.info(f"Computing atomic EMD")
    ref_mean, ref_covar = get_mean_covar(traj_aa[RAND1K].xyz)
    pred_mean, pred_covar = get_mean_covar(predtraj_aa.xyz)
    out["emd_mean"] = (np.square(ref_mean - pred_mean).sum(-1) ** 0.5) * 10
    try:
        out["emd_var"] = (
            np.trace(
                ref_covar + pred_covar - 2 * sqrtm(ref_covar @ pred_covar),
                axis1=1,
                axis2=2,
            )
            ** 0.5
        ) * 10
    except:
        out["emd_var"] = np.trace(ref_covar) ** 0.5 * 10

    logging.info(f"Analyzing SASA")
    sasa_thresh = 0.02
    ref_sasa = mdtraj.shrake_rupley(traj_aa[RAND1K], probe_radius=0.28)
    ref_sasa = condense_sidechain_sasas(ref_sasa, traj_aa.top)
    crystal_sasa = mdtraj.shrake_rupley(ref_aa, probe_radius=0.28)
    out["crystal_sasa"] = condense_sidechain_sasas(crystal_sasa, ref_aa.top)
    out["ref_sa_prob"] = (ref_sasa > sasa_thresh).mean(0)
    out["ref_mi_mat"] = sasa_mi(ref_sasa > sasa_thresh)

    # mdtraj's Shrake-Rupley calls exit() (uncatchable) on coincident atoms,
    # so drop such frames from the SASA computation only.
    sasa_frames = [
        i for i in range(predtraj_aa.n_frames)
        if not cKDTree(predtraj_aa.xyz[i]).query_pairs(1e-4)
    ]
    out["sasa_excluded_frames"] = predtraj_aa.n_frames - len(sasa_frames)
    if out["sasa_excluded_frames"]:
        logging.info(f"Excluding {out['sasa_excluded_frames']} frames with coincident atoms from SASA")
    if sasa_frames:
        pred_sasa = mdtraj.shrake_rupley(predtraj_aa[sasa_frames], probe_radius=0.28)
        pred_sasa = condense_sidechain_sasas(pred_sasa, predtraj_aa.top)
        out["pred_sa_prob"] = (pred_sasa > sasa_thresh).mean(0)
        out["pred_mi_mat"] = sasa_mi(pred_sasa > sasa_thresh)
    else:
        out["pred_sa_prob"] = None
        out["pred_mi_mat"] = None
    
    ref_distmat = np.linalg.norm(
        traj[RAND1].xyz[:, None, :] - traj[RAND1].xyz[:, :, None], axis=-1
    )
    pred_distmat = np.linalg.norm(
        predtraj.xyz[:, None, :] - predtraj.xyz[:, :, None], axis=-1
    )

    out["ref_contact_prob"] = (ref_distmat < 0.8).mean(0)
    out["pred_contact_prob"] = (pred_distmat < 0.8).mean(0)
    out["crystal_distmat"] = np.linalg.norm(
        ref.xyz[0, None, :] - ref.xyz[0, :, None], axis=-1
    )

    out["ref_mean_pairwise_rmsd"] = get_rmsds(
        traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True
    ).mean()
    out["pred_mean_pairwise_rmsd"] = get_rmsds(
        predtraj.xyz, predtraj.xyz, broadcast=True
    ).mean()

    out["ref_rms_pairwise_rmsd"] = (
        np.square(get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True)).mean()
        ** 0.5
    )
    out["pred_rms_pairwise_rmsd"] = (
        np.square(get_rmsds(predtraj.xyz, predtraj.xyz, broadcast=True)).mean() ** 0.5
    )

    out["ref_self_mean_pairwise_rmsd"] = get_rmsds(
        traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True
    ).mean()
    out["ref_self_rms_pairwise_rmsd"] = (
        np.square(get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True)).mean()
        ** 0.5
    )

    out["cosine_sim"] = (ref_pca.components_[0] * pred_pca.components_[0]).sum()

    def get_emd(ref_coords1, ref_coords2, pred_coords, seed_coords, K=None):
        if len(ref_coords1.shape) == 3:
            ref_coords1 = ref_coords1.reshape(ref_coords1.shape[0], -1)
            ref_coords2 = ref_coords2.reshape(ref_coords2.shape[0], -1)
            pred_coords = pred_coords.reshape(pred_coords.shape[0], -1)
            seed_coords = seed_coords.reshape(seed_coords.shape[0], -1)
        if K is not None:
            ref_coords1 = ref_coords1[:, :K]
            ref_coords2 = ref_coords2[:, :K]
            pred_coords = pred_coords[:, :K]
            seed_coords = seed_coords[:, :K]
        emd = {}
        emd["ref|ref mean"] = (
            (np.square(ref_coords1 - ref_coords1.mean(0)).sum(-1)).mean() ** 0.5
            / n_atoms**0.5
            * 10
        )

        distmat = np.square(ref_coords1[:, None] - ref_coords2[None]).sum(-1)
        distmat = distmat**0.5 / n_atoms**0.5 * 10
        emd["ref|ref2"] = get_wasserstein(distmat)
        emd["ref mean|ref2 mean"] = (
            np.square(ref_coords1.mean(0) - ref_coords2.mean(0)).sum() ** 0.5
            / n_atoms**0.5
            * 10
        )

        distmat = np.square(ref_coords1[:, None] - pred_coords[None]).sum(-1)
        distmat = distmat**0.5 / n_atoms**0.5 * 10
        emd["ref|prediction"] = get_wasserstein(distmat)
        emd["ref mean|prediction mean"] = (
            np.square(ref_coords1.mean(0) - pred_coords.mean(0)).sum() ** 0.5
            / n_atoms**0.5
            * 10
        )

        emd["ref|seed"] = (
            (np.square(ref_coords1 - seed_coords).sum(-1)).mean() ** 0.5
            / n_atoms**0.5
            * 10
        )
        emd["ref mean|seed"] = (
            (np.square(ref_coords1.mean(0) - seed_coords).sum(-1)).mean() ** 0.5
            / n_atoms**0.5
            * 10
        )

        emd["prediction|seed"] = (
            (np.square(pred_coords - seed_coords).sum(-1)).mean() ** 0.5
            / n_atoms**0.5
            * 10
        )
        emd["prediction|prediction mean"] = (
            (np.square(pred_coords - pred_coords.mean(0)).sum(-1)).mean() ** 0.5
            / n_atoms**0.5
            * 10
        )
        emd["prediction mean|seed"] = (
            (np.square(pred_coords.mean(0) - seed_coords).sum(-1)).mean() ** 0.5
            / n_atoms**0.5
            * 10
        )
        return emd

    K = 2
    out[f"EMD,ref"] = get_emd(
        ref_coords[RAND1],
        ref_coords[RAND2],
        pred_coords_ref_pca,
        seed_coords_ref_pca,
        K=K,
    )
    out[f"EMD,predicted"] = get_emd(
        ref_coords_pred_pca[RAND1],
        ref_coords_pred_pca[RAND2],
        pred_coords,
        seed_coords_pred_pca,
        K=K,
    )
    out[f"EMD,joint"] = get_emd(
        ref_coords_joint_pca[RAND1],
        ref_coords_joint_pca[RAND2],
        pred_coords_joint_pca,
        seed_coords_joint_pca,
        K=K,
    )
    out["nframes"] = predtraj.n_frames
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--bb_only", action="store_true")
    parser.add_argument("--ca_only", action="store_true")
    parser.add_argument("--max_pred_frames", type=int, default=None,
                        help="Subsample predicted trajectory to this many frames before analysis.")

    args = parser.parse_args()

    # Run analysis
    out = main(args)
    
    # Save output
    output_file = f"{args.output_dir}/{args.dataset}/{args.name}/{args.model}/analysis.pkl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logging.info(f"Writing to {os.path.abspath(output_file)}")
    with open(output_file, "wb") as f:
        f.write(pickle.dumps(out))
