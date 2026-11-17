from typing import Any
import tqdm
import numpy as np
import pickle
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os
import glob
import collections
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(levelname)s]: %(message)s"
)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")


def load_analysis_file(
    pred_dir: str, pattern: str = "analysis.pkl"
):
    """Load a single analysis.pkl file from given directory."""
    pkl_file = os.path.join(pred_dir, pattern)

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    return data


def correlations(a, b, prefix=""):
    a = np.array(a)
    b = np.array(b)

    # If a or b are constant, return NaNs
    if np.all(a == a[0]) or np.all(b == b[0]):
        return {
            prefix + "pearson": np.nan,
            prefix + "spearman": np.nan,
            prefix + "kendall": np.nan,
        }

    return {
        prefix + "pearson": scipy.stats.pearsonr(a, b)[0],
        prefix + "spearman": scipy.stats.spearmanr(a, b)[0],
        prefix + "kendall": scipy.stats.kendalltau(a, b)[0],
    }


def analyze_data(data):
    mi_mats = {}
    df = []
    for name, out in data.items():
        item = {
            "name": name,
            "ref_pairwise": out["ref_mean_pairwise_rmsd"],
            "pred_pairwise": out.get(
                "pred_mean_pairwise_rmsd", np.nan
            ),
            "cosine_sim": abs(out["cosine_sim"]),
            "emd_mean": np.square(out["emd_mean"]).mean() ** 0.5,
            "emd_var": np.square(out["emd_var"]).mean() ** 0.5,
            "pred_lddt_precision": out.get("pred_lddt_precision", np.nan),
            "pred_lddt_recall": out.get("pred_lddt_recall", np.nan),
            "pred_lddt_diversity": out.get("pred_lddt_diversity", np.nan),
            "ref_lddt_diversity": out.get("ref_lddt_diversity", np.nan),
            "mean_pred_lddt_to_ref": out.get("mean_pred_lddt_to_ref", np.nan),
            "mean_ref_lddt_to_ref": out.get("mean_ref_lddt_to_ref", np.nan),
        }
        item.update(correlations(out["pred_rmsf"], out["ref_rmsf"], prefix="rmsf_"))

        for emd_dict, emd_key in [
            (out["EMD,ref"], "ref"),
            (out["EMD,joint"], "joint"),
        ]:
            # Handle different key naming conventions
            ref_pred_key = None
            ref_mean_pred_mean_key = None

            for key in emd_dict.keys():
                if ("ref|" in key) and ("prediction" in key):
                    ref_pred_key = key
                if ("ref mean|" in key) and ("prediction mean" in key):
                    ref_mean_pred_mean_key = key

            if ref_pred_key and ref_mean_pred_mean_key:
                emd_val = emd_dict[ref_pred_key]
                emd_tr_val = emd_dict[ref_mean_pred_mean_key]
                item.update(
                    {
                        emd_key + "emd": emd_val,
                        emd_key + "emd_tr": emd_tr_val,
                        emd_key + "emd_int": (emd_val**2 - emd_tr_val**2) ** 0.5,
                    }
                )
            else:
                item.update(
                    {
                        emd_key + "emd": np.nan,
                        emd_key + "emd_tr": np.nan,
                        emd_key + "emd_int": np.nan,
                    }
                )

        crystal_contact_mask = out["crystal_distmat"] < 0.8
        ref_contact_prob = out["ref_contact_prob"]
        pred_contact_prob = out["pred_contact_prob"]

        if pred_contact_prob is not None:
            ref_transient_mask = (~crystal_contact_mask) & (ref_contact_prob > 0.1)
            pred_transient_mask = (~crystal_contact_mask) & (
                pred_contact_prob > 0.1
            )
            ref_weak_mask = crystal_contact_mask & (ref_contact_prob < 0.9)
            pred_weak_mask = crystal_contact_mask & (pred_contact_prob < 0.9)

            # assert (ref_weak_mask | pred_weak_mask).sum() > 0, "No weak contacts found!"
            # assert (ref_transient_mask | pred_transient_mask).sum() > 0, "No transient contacts found!"
            item.update(
                {
                    "weak_contacts_iou": (ref_weak_mask & pred_weak_mask).sum()
                    / (ref_weak_mask | pred_weak_mask).sum(),
                    "transient_contacts_iou": (
                        ref_transient_mask & pred_transient_mask
                    ).sum()
                    / (ref_transient_mask | pred_transient_mask).sum(),
                }
            )
        else:
            item.update(
                {
                    "weak_contacts_iou": np.nan,
                    "transient_contacts_iou": np.nan,
                }
            )

        sasa_thresh = 0.02
        buried_mask = out["crystal_sasa"][0] < sasa_thresh
        ref_sa_prob = out["ref_sa_prob"]
        pred_sa_prob = out["pred_sa_prob"]

        if pred_sa_prob is not None:
            ref_sa_mask = (ref_sa_prob > 0.1) & buried_mask
            pred_sa_mask = (pred_sa_prob > 0.1) & buried_mask

            item.update(
                {
                    "num_sasa": ref_sa_mask.sum(),
                    "sasa_iou": (ref_sa_mask & pred_sa_mask).sum()
                    / (ref_sa_mask | pred_sa_mask).sum(),
                }
            )
        else:
            item.update(
                {
                    "num_sasa": np.nan,
                    "sasa_iou": np.nan,
                }
            )

        ref_mi_mat = out["ref_mi_mat"]
        pred_mi_mat = out["pred_mi_mat"]

        if pred_mi_mat is not None:
            item.update(
                correlations(
                    ref_mi_mat.flatten(),
                    pred_mi_mat.flatten(),
                    prefix="exposon_mi_",
                )
            )
        else:
            item.update(
                {
                    "exposon_mi_pearson": np.nan,
                    "exposon_mi_spearman": np.nan,
                    "exposon_mi_kendall": np.nan,
                }
            )
        df.append(item)

    df = pd.DataFrame(df).set_index("name")

    all_ref_rmsf = []
    all_pred_rmsf = []

    for name in df.index:
        ref_rmsf = data[name].get("ref_rmsf", [])
        pred_rmsf = data[name].get("pred_rmsf", [])

        if len(ref_rmsf) > 0:
            all_ref_rmsf.extend(ref_rmsf)
        if len(pred_rmsf) > 0:
            all_pred_rmsf.extend(pred_rmsf)

    all_ref_rmsf = np.array(all_ref_rmsf)
    all_pred_rmsf = np.array(all_pred_rmsf)

    return all_ref_rmsf, all_pred_rmsf, df, data


def identify_high_rmsf_systems(datas, output_dir):
    """Identify and report systems with highest RMSF values and poorest correlations."""
    
    all_protein_stats = []
    
    for model_name, (ref_rmsf, pred_rmsf, df, data) in datas.items():
        for protein_name, row in df.iterrows():
            protein_ref_rmsf = data[protein_name]["ref_rmsf"]
            protein_pred_rmsf = data[protein_name]["pred_rmsf"]
            
            stats = {
                "Model": model_name,
                "Protein": protein_name,
                "Mean Ref RMSF": np.mean(protein_ref_rmsf),
                "Max Ref RMSF": np.max(protein_ref_rmsf),
                "Median Ref RMSF": np.median(protein_ref_rmsf),
                "Mean Pred RMSF": np.mean(protein_pred_rmsf),
                "Max Pred RMSF": np.max(protein_pred_rmsf),
                "Median Pred RMSF": np.median(protein_pred_rmsf),
                "RMSF Pearson": row["rmsf_pearson"],
                "RMSF Spearman": row["rmsf_spearman"],
                "Pairwise RMSD": row["pred_pairwise"],
                "Ref Pairwise RMSD": row["ref_pairwise"],
            }
            all_protein_stats.append(stats)
    
    stats_df = pd.DataFrame(all_protein_stats)
    
    # Sort by different criteria
    print("\n=== TOP 10 SYSTEMS BY MEAN REFERENCE RMSF ===")
    top_mean_rmsf = stats_df.nlargest(10, "Mean Ref RMSF")
    print(top_mean_rmsf[["Model", "Protein", "Mean Ref RMSF", "Mean Pred RMSF", "RMSF Pearson"]])
    
    print("\n=== TOP 10 SYSTEMS BY MAX REFERENCE RMSF ===")
    top_max_rmsf = stats_df.nlargest(10, "Max Ref RMSF")
    print(top_max_rmsf[["Model", "Protein", "Max Ref RMSF", "Max Pred RMSF", "RMSF Pearson"]])
    
    print("\n=== BOTTOM 10 SYSTEMS BY RMSF CORRELATION ===")
    bottom_corr = stats_df.nsmallest(10, "RMSF Pearson")
    print(bottom_corr[["Model", "Protein", "Mean Ref RMSF", "RMSF Pearson", "RMSF Spearman"]])
    
    print("\n=== TOP 10 MOST DYNAMIC SYSTEMS (Highest Pairwise RMSD) ===")
    top_dynamic = stats_df.nlargest(10, "Ref Pairwise RMSD")
    print(top_dynamic[["Model", "Protein", "Ref Pairwise RMSD", "Mean Ref RMSF", "RMSF Pearson"]])
    
    # Save full stats
    stats_df.to_csv(os.path.join(output_dir, "protein_rmsf_stats.csv"), index=False)
    print(f"\nFull protein statistics saved to {output_dir}/protein_rmsf_stats.csv")
    

def create_plots(plot_df, datas, output_dir: str):
    """Create comprehensive plots for the analysis results."""

    # Generate enough colors for all models
    n_models = len(plot_df.index)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / max(n_models - 1, 1)) for i in range(n_models)]

    # RMSD Analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for color, model in zip(colors, plot_df.index):
        axes[0].scatter(
            plot_df.loc[model]["Ref pairwise RMSD (all)"],
            plot_df.loc[model]["Pairwise RMSD (all)"],
            color=color,
            s=100,
            alpha=0.7,
            label=model,
        )

    # Diagonal line
    max_rmsd = max(
        plot_df["Ref pairwise RMSD (all)"].apply(max).max(),
        plot_df["Pairwise RMSD (all)"].apply(max).max(),
    )
    axes[0].plot([0, max_rmsd], [0, max_rmsd], "r--", alpha=0.7)
    axes[0].set_xlabel("Reference Pairwise RMSD (Å)")
    axes[0].set_ylabel("Predicted Pairwise RMSD (Å)")
    axes[0].set_title("Pairwise RMSD Comparison")
    axes[0].ticklabel_format(style="plain", axis="both")
    axes[0].grid(True, alpha=0.3)

    # --- RMSD correlation bar plot ---
    models = plot_df.index
    n_models = len(models)
    width = 0.15
    spacing = 1.2
    metric_positions = np.arange(3) * spacing  # 3 metrics

    metrics_rmsd = {
        "Pairwise RMSD r": "Pearson",
        "Pairwise RMSD rho": "Spearman",
        "Pairwise RMSD tau": "Kendall",
    }

    for i, model in enumerate(models):
        model_offset = (i - (n_models - 1) / 2) * width
        for j, metric_key in enumerate(metrics_rmsd.keys()):
            axes[1].bar(
                metric_positions[j] + model_offset,
                plot_df.loc[model, metric_key],
                width,
                label=model if j == 0 else "",
                color=colors[i],
                alpha=0.7,
            )

    axes[1].set_xticks(metric_positions)
    axes[1].set_xticklabels(metrics_rmsd.values())
    axes[1].set_ylabel("Pairwise RMSD Correlation")
    axes[1].set_title(r"Pairwise RMSD Correlation $\uparrow$")
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/rmsd_analysis.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close()

    # RMSF Analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- RMSF scatter ---
    for color, model in zip(colors, plot_df.index):
        axes[0].scatter(
            plot_df.loc[model]["Ref RMSF (all)"],
            plot_df.loc[model]["RMSF (all)"],
            color=color,
            s=100,
            alpha=0.3,
            label=model,
        )

    max_rmsf = max(
        plot_df["Ref RMSF (all)"].apply(max).max(),
        plot_df["RMSF (all)"].apply(max).max(),
    )
    axes[0].plot([0, max_rmsf], [0, max_rmsf], "r--", alpha=0.7)

    axes[0].set_xlabel("Reference RMSF (Å)")
    axes[0].set_ylabel("Predicted RMSF (Å)")
    axes[0].set_title("RMSF Comparison")
    axes[0].ticklabel_format(style="plain", axis="both")
    axes[0].grid(True, alpha=0.3)

    # --- RMSF correlations grouped bar plot ---
    models = plot_df.index
    n_models = len(models)
    width = 0.15
    spacing = 1.2
    metric_positions = np.arange(4) * spacing

    metrics = {
        "Global RMSF r": "Pearson \n(Global)",
        "Per target RMSF r": "Pearson\n(Per target)",
        "Per target RMSF rho": "Spearman\n(Per target)",
        "Per target RMSF tau": "Kendall\n(Per target)",
    }
    
    for i, model in enumerate(models):
        model_offset = (i - (n_models - 1) / 2) * width
        for j, metric_key in enumerate(metrics.keys()):
            axes[1].bar(
                metric_positions[j] + model_offset,
                plot_df.loc[model, metric_key],
                width,
                label=model if j == 0 else "",
                color=colors[i],
                alpha=0.7,
            )

    axes[1].set_xticks(metric_positions)
    axes[1].set_xticklabels(metrics.values())
    axes[1].set_ylabel("Correlation")
    axes[1].set_title(r"RMSF Correlation $\uparrow$")
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/rmsf_analysis.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close()
    
    # lDDT Analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # lDDT Precision
    valid_lddt_prec = plot_df["lDDT Precision"].dropna()
    valid_indices = [i for i, model in enumerate(plot_df.index) if model in valid_lddt_prec.index]
    axes[0, 0].bar(range(len(valid_lddt_prec)), valid_lddt_prec.values,
                color=[colors[i] for i in valid_indices], alpha=0.7)
    axes[0, 0].set_xticks(range(len(valid_lddt_prec)))
    axes[0, 0].set_xticklabels(valid_lddt_prec.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('lDDT Precision')
    axes[0, 0].set_title(r'lDDT Precision $\uparrow$')
    axes[0, 0].ticklabel_format(style='plain', axis='y')
    axes[0, 0].grid(True, alpha=0.3)

    # lDDT Recall
    valid_lddt_recall = plot_df["lDDT Recall"].dropna()
    valid_indices = [i for i, model in enumerate(plot_df.index) if model in valid_lddt_recall.index]
    axes[0, 1].bar(range(len(valid_lddt_recall)), valid_lddt_recall.values,
                color=[colors[i] for i in valid_indices], alpha=0.7)
    axes[0, 1].set_xticks(range(len(valid_lddt_recall)))
    axes[0, 1].set_xticklabels(valid_lddt_recall.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('lDDT Recall')
    axes[0, 1].set_title(r'lDDT Recall $\uparrow$')
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    axes[0, 1].grid(True, alpha=0.3)

    # lDDT Diversity comparison
    valid_pred_div = plot_df["lDDT Diversity"].dropna()
    valid_ref_div = plot_df["Ref lDDT Diversity"].dropna()
    common_div = valid_pred_div.index.intersection(valid_ref_div.index)

    # Get the predicted values and colors for the common entries
    predicted_values = valid_pred_div[common_div].values
    common_indices = [i for i, model in enumerate(plot_df.index) if model in common_div]
    plot_colors = [colors[i] for i in common_indices]
    plot_labels = list(common_div)

    # Check that reference diversity is consistent across common entries
    ref_values = valid_ref_div.loc[common_div]
    ref_value = ref_values.mean()

    x = np.arange(len(plot_labels))

    # 1. Plot the individual bars for Predicted diversity
    axes[1, 0].bar(x, predicted_values, color=plot_colors, alpha=0.7)

    # 2. Add a single horizontal line for the Reference diversity
    axes[1, 0].axhline(y=ref_value, color='k', linestyle='--',
                       label=f'Reference')

    # 3. Set labels and formatting
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(plot_labels, rotation=45, ha='right')
    axes[1, 0].set_ylabel('lDDT Diversity')
    axes[1, 0].set_title('lDDT Diversity')
    axes[1, 0].legend()
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    axes[1, 0].grid(True, alpha=0.3)

    # Mean lDDT to Reference
    valid_pred_mean_lddt = plot_df["Pred Mean lDDT to Ref"].dropna()
    valid_ref_mean_lddt = plot_df["Ref Mean lDDT to Ref"].dropna()
    common_mean_lddt = valid_pred_mean_lddt.index.intersection(valid_ref_mean_lddt.index)
    predicted_values = valid_pred_mean_lddt[common_mean_lddt].values
    common_indices = [i for i, model in enumerate(plot_df.index) if model in common_mean_lddt]
    plot_colors = [colors[i] for i in common_indices]
    plot_labels = list(common_mean_lddt)

    # Get the single reference value (can just take the first one)
    ref_values = valid_ref_mean_lddt.loc[common_mean_lddt]
    ref_value = ref_values.mean()

    # Plot the individual bars for Predicted diversity
    x = np.arange(len(plot_labels))
    axes[1, 1].bar(x, predicted_values, color=plot_colors, alpha=0.7)

    # Add a single horizontal line for the Reference diversity
    axes[1, 1].axhline(y=ref_value, color='k', linestyle='--',
                        label=f'Reference')

    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(plot_labels, rotation=45, ha='right')
    axes[1, 1].set_title(r'Mean lDDT to Reference Crystal')    
    axes[1, 1].set_ylabel('lDDT Score')
    axes[1, 1].legend()
    axes[1, 1].ticklabel_format(style='plain', axis='y')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/lddt_analysis.png", dpi=300, bbox_inches='tight',
            transparent=False, facecolor='white')
    plt.close()

    # 2. EMD and Dynamics Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # RMWD components
    valid_rmwd = plot_df['RMWD'].dropna()
    if len(valid_rmwd) > 0:
        valid_indices = [i for i, model in enumerate(plot_df.index) if model in valid_rmwd.index]
        axes[0, 0].bar(range(len(valid_rmwd)), valid_rmwd.values, 
                      color=[colors[i] for i in valid_indices], alpha=0.7)
        axes[0, 0].set_xticks(range(len(valid_rmwd)))
        axes[0, 0].set_xticklabels(valid_rmwd.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('RMWD (Å)')
        axes[0, 0].set_title(r'Root Mean Wasserstein Distance $\downarrow$')
        axes[0, 0].grid(True, alpha=0.3)
    
    # RMWD decomposition
    valid_trans = plot_df['RMWD trans'].dropna()
    valid_var = plot_df['RMWD var'].dropna()
    common_rmwd = valid_trans.index.intersection(valid_var.index)
    
    valid_trans = plot_df['RMWD trans'].dropna()
    valid_var = plot_df['RMWD var'].dropna()
    common_rmwd = valid_trans.index.intersection(valid_var.index)
    
    x = np.arange(len(common_rmwd))
    width = 0.35
    common_indices = [i for i, model in enumerate(plot_df.index) if model in common_rmwd]
    
    axes[0, 1].bar(x - width/2, valid_trans[common_rmwd].values, width, 
                    label='Translational', color=[colors[i] for i in common_indices], alpha=0.7)
    axes[0, 1].bar(x + width/2, valid_var[common_rmwd].values, width, 
                    label='Variational', color=[colors[i] for i in common_indices], alpha=0.5)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(common_rmwd, rotation=45, ha='right')
    axes[0, 1].set_ylabel('RMWD Component (Å)')
    axes[0, 1].set_title('RMWD Components')
    axes[0, 1].legend()
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    axes[0, 1].grid(True, alpha=0.3)

    valid_ref_pca = plot_df["Ref PCA W2"].dropna()
    valid_joint_pca = plot_df["Joint PCA W2"].dropna()
    common_pca = valid_ref_pca.index.intersection(valid_joint_pca.index)

    x = np.arange(len(common_pca))
    width = 0.35
    common_indices = [
        i for i, model in enumerate(plot_df.index) if model in common_pca
    ]

    axes[1, 0].bar(
        x - width / 2,
        valid_ref_pca[common_pca].values,
        width,
        label="W2 Distance using Reference PCA",
        color=[colors[i] for i in common_indices],
        alpha=0.7,
    )
    axes[1, 0].bar(
        x + width / 2,
        valid_joint_pca[common_pca].values,
        width,
        label="W2 Distance using Joint PCA",
        color=[colors[i] for i in common_indices],
        alpha=0.5,
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(common_pca, rotation=45, ha="right")
    axes[1, 0].set_ylabel("Wasserstein Distance (Å)")
    axes[1, 0].set_title(r"PCA-based Wasserstein Distances $\downarrow$")
    axes[1, 0].legend()
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    axes[1, 0].grid(True, alpha=0.3)

    # PC similarity
    valid_pc_sim = plot_df["PC sim > 0.5 %"].dropna()
    valid_indices = [
        i for i, model in enumerate(plot_df.index) if model in valid_pc_sim.index
    ]
    axes[1, 1].bar(
        range(len(valid_pc_sim)),
        valid_pc_sim.values,
        color=[colors[i] for i in valid_indices],
        alpha=0.7,
    )
    axes[1, 1].set_xticks(range(len(valid_pc_sim)))
    axes[1, 1].set_xticklabels(valid_pc_sim.index, rotation=45, ha="right")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].set_title(r"PC Similarity > 0.5 (%) $\uparrow$")
    axes[1, 1].ticklabel_format(style='plain', axis='y')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/dynamics_analysis.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close()

    # --- Contact IoU comparison ---
    fig, ax = plt.subplots(figsize=(8, 6))

    valid_weak = plot_df["Weak contacts J"].dropna()
    valid_transient = plot_df["Transient contacts J"].dropna()
    common_contacts = valid_weak.index.intersection(valid_transient.index)
    x = np.arange(len(common_contacts))
    width = 0.35
    common_indices = [i for i, m in enumerate(plot_df.index) if m in common_contacts]

    ax.bar(
        x - width / 2,
        valid_weak[common_contacts].values,
        width,
        label="Weak contacts",
        color=[colors[i] for i in common_indices],
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        valid_transient[common_contacts].values,
        width,
        label="Transient contacts",
        color=[colors[i] for i in common_indices],
        alpha=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(common_contacts, rotation=45, ha="right")
    ax.set_ylabel("Jaccard Index")
    ax.set_title(r"Contact Prediction Accuracy $\uparrow$")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/contacts_iou.png", dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()

    # --- Contact NaN percentages ---
    fig, ax = plt.subplots(figsize=(8, 6))

    print(plot_df["SASA nans"])
    valid_weak_nan = plot_df["Weak contacts nans"].dropna()
    valid_trans_nan = plot_df["Transient contacts nans"].dropna()
    valid_sasa_nan = plot_df["SASA nans"].dropna()
    common_nan = valid_weak_nan.index.intersection(valid_trans_nan.index).intersection(valid_sasa_nan.index)

    x = np.arange(len(common_nan))
    width = 0.35
    common_indices = [i for i, m in enumerate(plot_df.index) if m in common_nan]

    ax.bar(
        x - width / 2,
        valid_weak_nan[common_nan].values * 100,
        width,
        label="Weak contacts",
        color=[colors[i] for i in common_indices],
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        valid_trans_nan[common_nan].values * 100,
        width,
        label="Transient contacts",
        color=[colors[i] for i in common_indices],
        alpha=0.5,
    )
    ax.bar(
        x + width * 1.5,
        valid_sasa_nan[common_nan].values * 100,
        width,
        label="SASA",
        color=[colors[i] for i in common_indices],
        alpha=0.3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(common_nan, rotation=45, ha="right")
    ax.set_ylabel("Missing Data (%)")
    ax.set_title(r"Contact Analysis Coverage $\downarrow$")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/contacts_nan.png", dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()

    # --- Surface accessibility (SASA) ---
    fig, ax = plt.subplots(figsize=(8, 6))

    valid_sasa = plot_df["Exposed residue J"].dropna()
    valid_indices = [i for i, m in enumerate(plot_df.index) if m in valid_sasa.index]

    ax.bar(
        range(len(valid_sasa)),
        valid_sasa.values,
        color=[colors[i] for i in valid_indices],
        alpha=0.7,
    )

    ax.set_xticks(range(len(valid_sasa)))
    ax.set_xticklabels(valid_sasa.index, rotation=45, ha="right")
    ax.set_ylabel("Jaccard Index")
    ax.set_title(r"Exposed Residue Prediction Accuracy $\uparrow$")
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/surface_accessibility.png", dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()

    # --- MI matrix correlation ---
    fig, ax = plt.subplots(figsize=(8, 6))

    valid_mi = plot_df["Exposed MI matrix rho"].dropna()
    valid_indices = [i for i, m in enumerate(plot_df.index) if m in valid_mi.index]

    ax.bar(
        range(len(valid_mi)),
        valid_mi.values,
        color=[colors[i] for i in valid_indices],
        alpha=0.7,
    )

    ax.set_xticks(range(len(valid_mi)))
    ax.set_xticklabels(valid_mi.index, rotation=45, ha="right")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title(r"Mutual Information Matrix Correlation $\uparrow$")
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mi_matrix_correlation.png", dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()

    # Collect individual protein data across models
    all_protein_data = []
    for i, (model_name, (ref_rmsf, pred_rmsf, df, data)) in enumerate(datas.items()):
        for idx, row in df.iterrows():
            protein_data = {
                "Model": model_name,
                "Protein": idx,
                "RMSD_Ratio": row["pred_pairwise"] / row["ref_pairwise"],
                "RMSF_Corr": row["rmsf_pearson"],
                "Cosine_Sim": row["cosine_sim"],
                "RMWD": np.sqrt(row["emd_mean"] ** 2 + row["emd_var"] ** 2),
                "Weak_Contacts": row["weak_contacts_iou"],
                "Transient_Contacts": row["transient_contacts_iou"],
                "LDDT_Precision": row["pred_lddt_precision"],
                "LDDT_Recall": row["pred_lddt_recall"],
                "LDDT_Diversity": row["pred_lddt_diversity"],
                "Ref_LDDT_Diversity": row["ref_lddt_diversity"],
            }
            all_protein_data.append(protein_data)

    protein_df = pd.DataFrame(all_protein_data)

    # Distribution plots
    metrics_to_plot = {
        "RMSF_Corr": r"RMSF Correlation $\uparrow$",
        "RMSD_Ratio": r"RMSD Ratio",
        "Cosine_Sim": r"Cosine Similarity $\uparrow$",
        "RMWD": r"Root Mean Wasserstein Distance $\downarrow$",
        "Weak_Contacts": r"Weak Contacts IoU $\uparrow$",
        "Transient_Contacts": r"Transient Contacts IoU $\uparrow$",
        "LDDT_Precision": r"lDDT Precision $\uparrow$",
        "LDDT_Recall": r"lDDT Recall $\uparrow$",
        "LDDT_Diversity": r"lDDT Diversity",
    }

    # 4. Distribution CDF plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics_to_plot.items()):
        if i < len(axes):

            # Determine global min/max for aligned x-axes
            min_val = float("inf")
            max_val = float("-inf")

            for model in protein_df["Model"].unique():
                model_data = protein_df[protein_df["Model"] == model][metric].dropna()
                if len(model_data) > 0:
                    min_val = min(model_data.min(), min_val)
                    max_val = max(model_data.max(), max_val)

            # Plot CDFs
            for j, model in enumerate(protein_df["Model"].unique()):
                model_data = protein_df[protein_df["Model"] == model][metric].dropna()

                if len(model_data) == 0:
                    continue

                # Compute empirical CDF
                sorted_vals = np.sort(model_data)
                cdf_vals = np.linspace(0, 1, len(sorted_vals))

                axes[i].plot(
                    sorted_vals,
                    cdf_vals,
                    label=model,
                    color=colors[j],
                    alpha=0.8,
                    linewidth=2,
                )

            axes[i].set_xlim(min_val, max_val)
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel("CDF")
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)
            axes[i].ticklabel_format(style="plain", axis="both")

    # Remove unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].remove()

    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/protein_distributions.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close()

    print(f"All plots saved to {os.path.abspath(output_dir)}.")


def parse_params(dir_name: str) -> str:
    """Parse parameters from directory name for labeling."""
    params = dir_name.split("results_")[-1]
    params = params.split("_")
    param_str = ""
    for param in params:
        if "=" in param:
            key, value = param.split("=")
            if key == "s" or key == "sigma":
                param_str += f"$\\sigma={value}$, "
            elif key == "d" or key == "delta":
                param_str += f"$\\delta={value}$, "
            elif key == "f" or key == "friction":
                param_str += f"$\\gamma={value}$"
    return param_str


def format_model_name(model: str) -> str:
    return {
        'alphaflow_md_distilled': 'AlphaFlow MD Distilled',
        'alphaflow_md_base': 'AlphaFlow MD Base',
        'bioemu': 'BioEmu',
        'boltz2_diffusion': 'Boltz-2 Diffusion',
        'boltz2_diffusion_empty_msa': 'Boltz-2 (no MSA)',
        'boltz2_jump': 'Boltz-2 Jump',
        'boltz2_jump_msa': 'Boltz-2 Jump (MSA)',
        'boltz2_jump_empty_msa': 'Boltz-2 Jump (no MSA)',
        'boltz2_jump_msa_no_init': 'Boltz-2 Jump (MSA, no init)',
        'boltz2_jump_empty_msa_no_init': 'Boltz-2 Jump (no MSA, no init)',
        'boltz2_jump_empty_msa_boltz2_init': 'Boltz-2 Jump (no MSA, Boltz-2 init)',
        'boltz2_jump_msa_d1': 'Boltz-2 Jump (MSA, δ=1)',
        'boltz2_jump_empty_msa_d1': 'Boltz-2 Jump (no MSA, δ=1)',
        'boltz2_jump_msa_no_init_d1': 'Boltz-2 Jump (MSA, no init, δ=1)',
        'boltz2_jump_empty_msa_no_init_d1': 'Boltz-2 Jump (no MSA, no init, δ=1)',
        'boltz2_jump_empty_msa_boltz2_init_d1': 'Boltz-2 Jump (no MSA, Boltz-2 init, δ=1)',
        'boltz2_jump_msa_d2_100': 'Boltz-2 Jump (MSA, δ=2, 100fr)',
        'boltz2_jump_empty_msa_d2_100': 'Boltz-2 Jump (no MSA, δ=2, 100fr)',
        'boltz2_jump_msa_no_init_d2_100': 'Boltz-2 Jump (MSA, no init, δ=2, 100fr)',
        'boltz2_jump_empty_msa_no_init_d2_100': 'Boltz-2 Jump (no MSA, no init, δ=2, 100fr)',
        'boltz2_jump_empty_msa_boltz2_init_d2_100': 'Boltz-2 Jump (no MSA, Boltz-2 init, δ=2, 100fr)',
        'boltz2_noised_seed_sde': 'Noised Seed SDE',
        'boltz2_noised_seed_ode': 'Noised Seed ODE',
        'boltz2_noised_seed_one_step': 'Noised Seed (1-step)',
        'boltz2_msa_subsample_32': 'Boltz-2 MSA Subsample 32',
        'boltz2_msa_subsample_64': 'Boltz-2 MSA Subsample 64',
        'boltz2_full_msa': 'Boltz-2 Full MSA',
        'boltz2_jump_empty_msa_boltz2_init_d1_v2': 'Boltz-2 Jump (no MSA, Boltz-2 + MSA init, δ=1)',
        'boltz2_noised_seed_sde_v2': 'Noised Seed SDE (fixed)',
        'boltz2_noised_seed_ode_v2': 'Noised Seed ODE (fixed)',
        'boltz2_jump_msa_d1_rep1': 'Boltz-2 Jump (MSA, δ=1, rep 1)',
        'boltz2_jump_msa_d1_rep2': 'Boltz-2 Jump (MSA, δ=1, rep 2)',
        'boltz2_walk_denoise_n10': 'Boltz-2 Walk-Denoise (MSA, δ=1, N=10)',
        'boltz2_walk_denoise_n53': 'Boltz-2 Walk-Denoise (MSA, δ=1, N=53)',
    }.get(model, model)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ensemble results and generate comprehensive plots"
    )
    parser.add_argument(
        "--predictions_root_dir",
        type=str,
        required=True,
        help="Root directory containing model subdirectories with analysis.pkl files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for plots and tables",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/analysis.pkl",
        help="Pattern to match analysis files",
    )
    args = parser.parse_args()

    # Define desired order up front so we can filter during loading
    desired_order = [
        'Boltz-2 Jump',
        'Boltz-2 Jump (MSA, δ=1)',
        'Boltz-2 Jump (MSA, δ=2, 100fr)',
        'Boltz-2 Jump (no MSA, δ=1)',
        'Boltz-2 Jump (MSA, no init, δ=1)',
        'Boltz-2 Jump (no MSA, no init, δ=1)',
        'Boltz-2 Jump (no MSA, Boltz-2 init, δ=1)',
        'Boltz-2 (no MSA)',
        'Noised Seed SDE',
        'Noised Seed ODE',
        'Noised Seed (1-step)',
        'Boltz-2 Full MSA',
        'Boltz-2 MSA Subsample 32',
        'Boltz-2 MSA Subsample 64',
        'AlphaFlow MD Base',
        'AlphaFlow MD Distilled',
        'BioEmu',
        'Boltz-2 Jump (no MSA, Boltz-2 + MSA init, δ=1)',
        'Noised Seed SDE (fixed)',
        'Noised Seed ODE (fixed)',
        'Boltz-2 Jump (MSA, δ=1, rep 1)',
        'Boltz-2 Jump (MSA, δ=1, rep 2)',
        'Boltz-2 Walk-Denoise (MSA, δ=1, N=10)',
        'Boltz-2 Walk-Denoise (MSA, δ=1, N=53)',
    ]

    # Load all analysis files using directory structure
    all_results = collections.defaultdict(dict)
    for system in os.listdir(args.predictions_root_dir):

        # if "1a62_A" not in pred_dir:
        #     continue

        # if "1ef1_D" not in pred_dir:
        #     continue

        # if "s=1.0" in pred_dir:
        #     continue

        system_dir = os.path.join(args.predictions_root_dir, system)
        for model in sorted(os.listdir(system_dir)):
            model_dir = os.path.join(system_dir, model)
            pkl = os.path.join(model_dir, "analysis.pkl")
            if not os.path.exists(pkl):
                continue
            result = load_analysis_file(model_dir)
            key = format_model_name(model)
            all_results[key].update({
                system: result
            })

    print(f"Loaded results for {len(all_results)} models.")
    
    # Use the intersection of systems across desired models only
    desired_results = {k: v for k, v in all_results.items() if k in desired_order}
    all_systems = None
    for key, results in desired_results.items():
        systems = set(results.keys())
        all_systems = systems if all_systems is None else all_systems & systems
    print(f"All models have results for {len(all_systems)} systems.")

    # Only include models that actually have results
    desired_order = [m for m in desired_order if m in all_results]
    datas = {}
    for key in desired_order:
        results = all_results[key]
        print(f"Processing {key} with {len(results)} systems.")
        datas[key] = analyze_data(results)

    # Generate summary table
    summary_df = []
    for key in datas:
        ref_rmsf, pred_rmsf, df, data = datas[key]

        global_rmsf_r = scipy.stats.pearsonr(ref_rmsf, pred_rmsf)[0]
        ref_rmsf_median = np.median(ref_rmsf)
        pred_rmsf_median = np.median(pred_rmsf)
        # print(df.pred_lddt_diversity, df.ref_lddt_diversity, df.mean_pred_lddt_to_ref.astype(float))
        
        rmsd_correlations = correlations(df.ref_pairwise, df.pred_pairwise)
        summary_df.append(
            {
                "path": key,
                "count": len(df),
                "Ref pairwise RMSD (all)": df.ref_pairwise.to_list(),
                "Pairwise RMSD (all)": df.pred_pairwise.to_list(),
                "Ref pairwise RMSD": df.ref_pairwise.median(),
                "Pairwise RMSD": df.pred_pairwise.median(),
                "Pairwise RMSD r": rmsd_correlations["pearson"],
                "Pairwise RMSD rho": rmsd_correlations["spearman"],
                "Pairwise RMSD tau": rmsd_correlations["kendall"],
                "Ref RMSF (all)": ref_rmsf,
                "RMSF (all)": pred_rmsf,
                "Ref RMSF": ref_rmsf_median,
                "RMSF": pred_rmsf_median,
                "Global RMSF r": global_rmsf_r,
                "Per target RMSF r": df.rmsf_pearson.median(),
                "Per target RMSF rho": df.rmsf_spearman.median(),
                "Per target RMSF tau": df.rmsf_kendall.median(),
                "RMWD": np.sqrt(df.emd_mean**2 + df.emd_var**2).median(),
                "RMWD trans": df.emd_mean.median(),
                "RMWD var": df.emd_var.median(),
                "Ref PCA W2": df.refemd.median(),
                "Joint PCA W2": df.jointemd.median(),
                "PC sim > 0.5 %": (df.cosine_sim > 0.5).mean() * 100,
                "Weak contacts J": df.weak_contacts_iou.median(),
                "Weak contacts nans": df.weak_contacts_iou.isna().mean(),
                "Transient contacts J": df.transient_contacts_iou.median(),
                "Transient contacts nans": df.transient_contacts_iou.isna().mean(),
                "SASA nans": df.sasa_iou.isna().mean(),
                "Exposed residue J": df.sasa_iou.median(),
                "Exposed MI matrix rho": df.exposon_mi_spearman.median(),
                "lDDT Precision": df.pred_lddt_precision.median(),
                "lDDT Recall": df.pred_lddt_recall.median(),
                "lDDT Diversity": df.pred_lddt_diversity.median(),
                "Ref lDDT Diversity": df.ref_lddt_diversity.median(),
                "Pred Mean lDDT to Ref": df.mean_pred_lddt_to_ref.astype(float).median(),
                "Ref Mean lDDT to Ref": df.mean_ref_lddt_to_ref.astype(float).median(),
            }
        )

    summary_df = pd.DataFrame(summary_df).set_index("path")
    # print("\n=== SUMMARY TABLE ===")
    # print(summary_df.round(2).T)

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # identify_high_rmsf_systems(datas, output_dir)
    # return 

    # Save summary table, excluding lists
    summary_df_no_lists = summary_df.drop(columns=[
        "Ref pairwise RMSD (all)",
        "Pairwise RMSD (all)",
        "Ref RMSF (all)",
        "RMSF (all)",
    ])
    summary_df_no_lists.to_csv(os.path.join(output_dir, "summary_table.csv"))

    # Generate plots
    create_plots(summary_df, datas, output_dir)


if __name__ == "__main__":
    main()
