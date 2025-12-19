import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from glm_pca import fit_binomial_glm_pca_earlystop, fit_binomial_glm_pca_earlystop_torch


# ---------- Helper: PCA on rows ----------
def pca_scores(X: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k]


def map_labels_to_ints(labels):
    """Map categorical labels -> integer codes, plus return unique order."""
    labels = np.asarray(labels).astype(str)
    uniq = np.unique(labels)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    ints = np.array([mapping[lab] for lab in labels], dtype=int)
    return ints, uniq, mapping


# ---------- Distance helpers for intra/inter ----------
def pairwise_euclidean_distances(X: np.ndarray) -> np.ndarray:
    """
    Return condensed vector of all pairwise Euclidean distances between rows of X
    (upper triangle of full distance matrix).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    m = n * (n - 1) // 2
    out = np.empty(m, dtype=float)
    idx = 0
    for i in range(n - 1):
        diff = X[i + 1:] - X[i]
        di = np.sqrt(np.sum(diff * diff, axis=1))
        out[idx: idx + di.size] = di
        idx += di.size
    return out


def intra_inter_distance_stats(embedding: np.ndarray, labels: np.ndarray) -> dict:
    """
    For each group c in labels:
      - mean intra-group distance
      - mean inter-group distance (c vs all others)
      - ratio inter / intra
    """
    embedding = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)

    stats = {}
    for c in unique:
        mask_c = labels == c
        mask_other = ~mask_c
        X_c = embedding[mask_c]
        X_other = embedding[mask_other]

        # Intra-group distances
        if X_c.shape[0] < 2:
            intra = np.nan
        else:
            intra = float(pairwise_euclidean_distances(X_c).mean())

        # Inter-group distances
        if X_c.shape[0] == 0 or X_other.shape[0] == 0:
            inter = np.nan
        else:
            diff = X_c[:, None, :] - X_other[None, :, :]
            dmat = np.sqrt(np.sum(diff * diff, axis=2))
            inter = float(dmat.mean())

        if np.isfinite(intra) and intra > 0 and np.isfinite(inter):
            ratio = inter / intra
        else:
            ratio = np.nan

        stats[str(c)] = {
            "n_cells": int(X_c.shape[0]),
            "intra_mean": intra,
            "inter_mean": inter,
            "ratio_inter_over_intra": float(ratio) if np.isfinite(ratio) else np.nan,
        }

    return stats


# ----------------------------
# Paths
# ----------------------------
raw_dir = "raw_data"   # adjust if needed

mc_path   = os.path.join(raw_dir, "mc_raw.h5ad")
cov_path  = os.path.join(raw_dir, "cov_raw.h5ad")
meta_path = os.path.join(raw_dir, "snmct_xgboost_v7.csv")


# ----------------------------
# Load MC / COV
# ----------------------------
mc_adata = sc.read_h5ad(mc_path)
cov_adata = sc.read_h5ad(cov_path)

assert mc_adata.shape == cov_adata.shape
n_cells, n_loci = mc_adata.shape
print(f"MC/COV loaded: {n_cells} cells × {n_loci} loci")

X_mc  = np.asarray(mc_adata.X,  dtype=np.float64)
X_cov = np.asarray(cov_adata.X, dtype=np.float64)


# ----------------------------
# Load metadata (tab-separated)
# ----------------------------
meta = pd.read_csv(meta_path, sep="\t")
print("Metadata columns:", meta.columns.tolist())
# expected: ['Unnamed: 0', 'donor', 'time_num', 'pred_coarse', 'coarse_donor']

csv_ids = meta["Unnamed: 0"].astype(str).values
id_to_donor       = dict(zip(csv_ids, meta["donor"].astype(str).values))
id_to_pred_coarse = dict(zip(csv_ids, meta["pred_coarse"].astype(str).values))
id_to_coarse      = dict(zip(csv_ids, meta["coarse_donor"].astype(str).values))

if "cell" not in mc_adata.obs.columns:
    raise ValueError("mc_adata.obs does not contain a 'cell' column; cannot align metadata.")

cell_ids = mc_adata.obs["cell"].astype(str).values

donor_labels       = np.array([id_to_donor.get(cid, "unknown")       for cid in cell_ids], dtype=object)
pred_coarse_labels = np.array([id_to_pred_coarse.get(cid, "unknown") for cid in cell_ids], dtype=object)
coarse_labels      = np.array([id_to_coarse.get(cid, "unknown")      for cid in cell_ids], dtype=object)

print("Unique donor labels:",        np.unique(donor_labels))
print("Unique pred_coarse labels:",  np.unique(pred_coarse_labels))
print("Unique coarse_donor labels:", np.unique(coarse_labels))


# ----------------------------
# Optional downsampling
# ----------------------------
target_cells = None   # None = use all cells
target_loci  = None   # None = use all loci

target_cells = 4000
target_loci  = 8000

rng = np.random.default_rng(0)

cell_idx = (
    np.arange(n_cells)
    if target_cells is None or target_cells >= n_cells
    else rng.choice(n_cells, size=target_cells, replace=False)
)

locus_idx = (
    np.arange(n_loci)
    if target_loci is None or target_loci >= n_loci
    else rng.choice(n_loci, size=target_loci, replace=False)
)

mc  = X_mc[cell_idx][:, locus_idx]
cov = X_cov[cell_idx][:, locus_idx]

donor_sub       = donor_labels[cell_idx]
pred_coarse_sub = pred_coarse_labels[cell_idx]
coarse_sub      = coarse_labels[cell_idx]

np.savez("metadata_downsampled_for_snaps.npz",
         pred_coarse_sub=pred_coarse_sub,
         donor_sub=donor_sub,
         coarse_sub=coarse_sub)



print(f"Downsampled: {mc.shape[0]} cells × {mc.shape[1]} loci")
print("Subsample donors:",        np.unique(donor_sub))
print("Subsample pred_coarse:",   np.unique(pred_coarse_sub))
print("Subsample coarse_donor:",  np.unique(coarse_sub))


# ----------------------------
# Run GLM-PCA (early stop)
# ----------------------------
k_latent    = 10
print_every = 200

U_glm, V_glm, alpha_glm, history_glm, masks_glm = fit_binomial_glm_pca_earlystop(
    mc,
    cov,
    n_components=k_latent,
    max_iter=2000,
    # max_iter=1000,
    lr=5e-3,
    l2_reg=1e-3,
    val_frac=0.1,
    patience=20,
    beta1=0.9,
    beta2=0.99,
    adam_eps=1e-8,
    print_every=print_every,
    seed=0,
)

# U_glm, V_glm, alpha_glm, history_glm, masks_glm = fit_binomial_glm_pca_earlystop_torch(
#     mc,
#     cov,
#     n_components=k_latent,
#     max_iter=4000,
#     lr=5e-3,
#     l2_reg=1e-3,
#     val_frac=0.1,
#     patience=20,
#     beta1=0.9,
#     beta2=0.99,
#     adam_eps=1e-8,
#     print_every=200,
#     seed=0,
#     device="mps",      # <-- GPU on your M1 Pro
#     dtype="float32",
# )



# ----------------------------
# Precompute label → color mappings
# ----------------------------
donor_ints,  donor_uniq,  donor_map  = map_labels_to_ints(donor_sub)
pred_ints,   pred_uniq,   pred_map   = map_labels_to_ints(pred_coarse_sub)
coarse_ints, coarse_uniq, coarse_map = map_labels_to_ints(coarse_sub)


def add_categorical_legend(ax, label_to_int, title):
    handles = []
    for lab, idx in label_to_int.items():
        color = plt.cm.tab20(idx % 20)
        h = plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=5,
            color=color,
        )
        handles.append(h)
    ax.legend(
        handles,
        list(label_to_int.keys()),
        title=title,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
    )


# ----------------------------
# Panels per iteration + intra/inter metrics
# ----------------------------
output_dir = "glmpca_real_panels"
os.makedirs(output_dir, exist_ok=True)

iters_hist = history_glm["iters"]
U_snaps    = history_glm["U_snapshots"]




np.savez(
    "glm_pca_snaps.npz",
    iters_hist=np.array(iters_hist, dtype=int),
    U_snaps=np.array(U_snaps, dtype=object)  # object array of snapshots
)


k_pc   = 10  # number of dims from U to use (for PCA + UMAP)
n_iter = len(iters_hist)
print(f"Creating panel figures and intra/inter metrics for {n_iter} iterations...")

metrics_rows = []  # will store intra/inter stats per iteration × coarse_donor

# ----------------------------
# Build combined legend ONCE
# ----------------------------
def legend_handles(mapping):
    handles, labels = [], []
    for lab, idx in mapping.items():
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=5,
                color=plt.cm.tab20(idx % 20),
            )
        )
        labels.append(str(lab))
    return handles, labels

donor_h, donor_l   = legend_handles(donor_map)
pred_h, pred_l     = legend_handles(pred_map)
coarse_h, coarse_l = legend_handles(coarse_map)

all_handles = donor_h + pred_h + coarse_h
all_labels = (
    [f"[donor] {x}" for x in donor_l] +
    [f"[pred] {x}" for x in pred_l] +
    [f"[coarse] {x}" for x in coarse_l]
)

# ----------------------------
# Main loop
# ----------------------------
for it, U_snap in zip(iters_hist, U_snaps):
    # Orthonormalize snapshot U
    U_snap_orth = pca_scores(U_snap, k_pc)
    pcs2 = U_snap_orth[:, :2]

    # Intra/inter stats (coarse_donor)
    stats = intra_inter_distance_stats(pcs2, coarse_sub)
    for group_name, s in stats.items():
        metrics_rows.append({
            "iter": it,
            "coarse_donor": group_name,
            "n_cells": s["n_cells"],
            "intra_mean": s["intra_mean"],
            "inter_mean": s["inter_mean"],
            "ratio_inter_over_intra": s["ratio_inter_over_intra"],
        })

    # Build AnnData for UMAP
    ad = sc.AnnData(X=U_snap_orth.copy())
    ad.obs["donor"]        = pd.Categorical(donor_sub)
    ad.obs["pred_coarse"]  = pd.Categorical(pred_coarse_sub)
    ad.obs["coarse_donor"] = pd.Categorical(coarse_sub)

    sc.pp.neighbors(ad, n_neighbors=15, random_state=0)
    sc.tl.umap(ad, random_state=0)
    umap = ad.obsm["X_umap"]

    # ---- Figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"GLM-PCA iteration {it}", fontsize=14)

    # Row 0: PCs
    axes[0,0].scatter(pcs2[:,0], pcs2[:,1], c=donor_ints, cmap="tab20", s=8, alpha=0.7)
    axes[0,0].set_title("PC1 vs PC2 (donor)")
    axes[0,0].set_xlabel("PC1"); axes[0,0].set_ylabel("PC2")
    axes[0,0].grid(alpha=0.3)

    axes[0,1].scatter(pcs2[:,0], pcs2[:,1], c=pred_ints, cmap="tab20", s=8, alpha=0.7)
    axes[0,1].set_title("PC1 vs PC2 (pred_coarse)")
    axes[0,1].set_xlabel("PC1"); axes[0,1].set_ylabel("PC2")
    axes[0,1].grid(alpha=0.3)

    axes[0,2].scatter(pcs2[:,0], pcs2[:,1], c=coarse_ints, cmap="tab20", s=8, alpha=0.7)
    axes[0,2].set_title("PC1 vs PC2 (coarse_donor)")
    axes[0,2].set_xlabel("PC1"); axes[0,2].set_ylabel("PC2")
    axes[0,2].grid(alpha=0.3)

    # Row 1: UMAP
    axes[1,0].scatter(umap[:,0], umap[:,1], c=donor_ints, cmap="tab20", s=8, alpha=0.7)
    axes[1,0].set_title("UMAP (donor)")
    axes[1,0].set_xlabel("UMAP1"); axes[1,0].set_ylabel("UMAP2")
    axes[1,0].grid(alpha=0.3)

    axes[1,1].scatter(umap[:,0], umap[:,1], c=pred_ints, cmap="tab20", s=8, alpha=0.7)
    axes[1,1].set_title("UMAP (pred_coarse)")
    axes[1,1].set_xlabel("UMAP1"); axes[1,1].set_ylabel("UMAP2")
    axes[1,1].grid(alpha=0.3)

    axes[1,2].scatter(umap[:,0], umap[:,1], c=coarse_ints, cmap="tab20", s=8, alpha=0.7)
    axes[1,2].set_title("UMAP (coarse_donor)")
    axes[1,2].set_xlabel("UMAP1"); axes[1,2].set_ylabel("UMAP2")
    axes[1,2].grid(alpha=0.3)

    # ---- ONE combined legend (no overlap) ----
    fig.legend(
        all_handles, all_labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=False,
        ncol=2,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    plt.tight_layout(rect=[0, 0, 0.78, 0.96])

    fname = os.path.join(output_dir, f"glmpca_panels_iter_{it:05d}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)

print(f"Saved panel figures for {n_iter} iterations in: {output_dir}")



# ----------------------------
# Save intra/inter metrics to CSV
# ----------------------------
metrics_df = pd.DataFrame(metrics_rows)
metrics_csv_path = "glmpca_coarse_donor_intra_inter.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved coarse_donor intra/inter metrics to: {metrics_csv_path}")

# ----------------------------
# Plots: ratio_inter_over_intra vs iteration
#   1) per donor
#   2) per cell type (from coarse_donor prefix)
# ----------------------------

# Split coarse_donor into cell_type and donor
metrics_df["cell_type"] = metrics_df["coarse_donor"].str.split("_").str[0]
metrics_df["donor"] = metrics_df["coarse_donor"].str.split("_").str[1]

def weighted_mean_and_se(values, weights):
    """
    Compute weighted mean and SE, ignoring NaNs in values.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values)
    if not mask.any():
        return np.nan, np.nan

    values = values[mask]
    weights = weights[mask]

    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    se = np.sqrt(var) / np.sqrt(weights.sum()) if weights.sum() > 0 else 0.0
    return mean, se


# ---------- Aggregate per donor per iteration ----------
donor_summary_rows = []
for (it, donor), sub in metrics_df.groupby(["iter", "donor"]):
    mean, se = weighted_mean_and_se(
        sub["ratio_inter_over_intra"],
        sub["n_cells"]
    )
    donor_summary_rows.append({
        "iter": it,
        "donor": donor,
        "mean": mean,
        "se": se,
    })

donor_summary = pd.DataFrame(donor_summary_rows).sort_values(["donor", "iter"])

# ---------- Aggregate per cell type per iteration ----------
celltype_summary_rows = []
for (it, ct), sub in metrics_df.groupby(["iter", "cell_type"]):
    mean, se = weighted_mean_and_se(
        sub["ratio_inter_over_intra"],
        sub["n_cells"]
    )
    celltype_summary_rows.append({
        "iter": it,
        "cell_type": ct,
        "mean": mean,
        "se": se,
    })

celltype_summary = pd.DataFrame(celltype_summary_rows).sort_values(["cell_type", "iter"])

# ----------------------------
# Plot 1: donor curves across iterations
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for donor, sub in donor_summary.groupby("donor"):
    ax.errorbar(
        sub["iter"],
        sub["mean"],
        yerr=sub["se"],
        marker="o",
        linestyle="-",
        label=str(donor),
    )

ax.set_xlabel("Iteration")
ax.set_ylabel("Mean ratio_inter_over_intra")
ax.set_title("Mean inter / intra distance by donor across iterations")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
fig.tight_layout()

donor_plot_path = os.path.join(output_dir, "ratio_inter_over_intra_by_donor_vs_iter.png")
fig.savefig(donor_plot_path, dpi=150)
print(f"Saved donor plot to: {donor_plot_path}")

# ----------------------------
# Plot 2: cell-type curves across iterations
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for ct, sub in celltype_summary.groupby("cell_type"):
    ax.errorbar(
        sub["iter"],
        sub["mean"],
        yerr=sub["se"],
        marker="o",
        linestyle="-",
        label=str(ct),
    )

ax.set_xlabel("Iteration")
ax.set_ylabel("Mean ratio_inter_over_intra")
ax.set_title("Mean inter / intra distance by cell type across iterations")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
fig.tight_layout()

celltype_plot_path = os.path.join(output_dir, "ratio_inter_over_intra_by_celltype_vs_iter.png")
fig.savefig(celltype_plot_path, dpi=150)
print(f"Saved cell-type plot to: {celltype_plot_path}")
