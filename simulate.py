from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pprint import pprint
Array = np.ndarray

# ----------------------------
# Config & Result containers
# ----------------------------

@dataclass
class SimConfig:
    n_cells: int = 1000
    n_loci: int = 2000
    # n_cells: int = 5000
    # n_loci: int = 5000
    celltype1_fraction: float = 0.7       # asymmetry: fraction of type 1
    diff_fraction: float = 0.3            # fraction of loci that differ between types

    # Baseline (shared) beta distribution (U-shaped if a,b<1)
    beta_1: float = 0.3
    beta_2: float = 0.3

    # Difference magnitude between types (logit-scale SD)
    diff_sigma: float = 0.12              # ~0.1–0.3 reasonable

    # Optional per-cell jitter around locus×type betas (logit scale)
    cell_jitter_sd: float = 0.15          # 0 disables jitter

    # Read depth: Zero-Inflated Negative Binomial via (mean & size)
    # These defaults are taken from your empirical ZINB fit:
    #   mu_cov ≈ 27.3, r_cov ≈ 0.96, pi_zero ≈ 0.136
    nb_mean: float = 27.29               # per-entry coverage mean
    nb_size: float = 0.956               # NB "size" parameter r_cov (~0.9557)
    zero_frac: float = 0.136               # small extra zeros; set to 0 to disable

    # Per-cell library size variation (log-normal multiplicative factor)
    # L_i ~ LogNormal(liblog_mu, liblog_sd)
    # Defaults: sd ~1.2 to match real per-cell dispersion; mu sets mean to 1.
    liblog_mu: float = -5e-07
    liblog_sd: float = 0.001

    # Per-locus coverage variation (log-normal multiplicative factor)
    # S_j ~ LogNormal(locus_log_mu, locus_log_sd) applied to NB mean per locus.
    # Defaults: sd ~0.8 to create a heavy-tailed per-bin spread; mu keeps mean ~1.
    locus_log_mu: float = -5.0e-07
    locus_log_sd: float = 0.001

    # Random seed
    seed: Optional[int] = 1

@dataclass
class SimulationResult:
    B_true: Array        # (n_cells, n_loci) true per-cell methylation fractions
    Z_true: Array        # (n_cells, n_loci) logits of B_true
    cov: Array           # (n_cells, n_loci) read depths (int >= 0)
    mc: Array            # (n_cells, n_loci) methylated counts (0..N)
    cell_type: Array     # (n_cells,) values in {0,1}
    is_diff: Array       # (n_loci,) boolean indicating differential loci

# ----------------------------
# Utilities
# ----------------------------

def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))

def logit_clip(p: Array, eps: float = 1e-6) -> Array:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _nbinom_mean_size(
    rng: np.random.Generator,
    mean: Union[Array, float],
    size: float,
    shape: Optional[Tuple[int, int]] = None,
) -> Array:
    """
    Draw from NB with parameters (mean, size).

    Supports:
    - scalar mean + shape       -> NB(mean, size) broadcast to `shape`
    - array mean (mu_ij)        -> NB(mean_ij, size) elementwise (shape inferred)

    numpy uses (n, p): mean = n(1-p)/p.
    Convert: p = size / (size + mean), n = size.
    """
    r = float(size)
    mean_arr = np.asarray(mean, dtype=float)

    # Infer shape if not provided
    if shape is None:
        shape = mean_arr.shape

    if mean_arr.shape == ():  # scalar mean
        mu = float(mean_arr)
        p = r / (r + mu)
        return rng.negative_binomial(n=r, p=p, size=shape).astype(np.int32)
    else:
        if shape != mean_arr.shape:
            raise ValueError(f"shape {shape} does not match mean.shape {mean_arr.shape}")
        mu = mean_arr
        p = r / (r + mu)
        return rng.negative_binomial(n=r, p=p, size=shape).astype(np.int32)

def _apply_diff_logit(base: Array, rng: np.random.Generator, sigma: float) -> Array:
    """
    Apply symmetric normal shift on the *logit* scale to produce type-B betas.
    base: 1D array of baseline betas (0,1) for type A at differential loci.
    """
    z = logit_clip(base)
    dz = rng.normal(loc=0.0, scale=sigma, size=base.shape)
    return _sigmoid(z + dz)

# ----------------------------
# Main simulator
# ----------------------------

def estimate_beta_prior_per_cell(mc: Array, cov: Array, eps: float = 1e-8):
    """
    Estimate per-cell Beta prior parameters (alpha_i, beta_i) using
    method-of-moments, following Liu et al. (Nature 2021).

    For each cell i, compute the mean m_i and variance v_i of raw
    methylation levels mc_ij / cov_ij over loci with cov_ij > 0, then:

        s_i = alpha_i + beta_i = m_i (1 - m_i) / v_i - 1
        alpha_i = m_i * s_i
        beta_i  = (1 - m_i) * s_i

    We add small epsilons / clamps for numerical stability.
    """
    mc = mc.astype(float)
    cov = cov.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = mc / cov
    frac[cov <= 0] = np.nan  # ignore zero-coverage loci

    # Per-cell mean and variance (ignoring NaNs)
    m = np.nanmean(frac, axis=1)
    v = np.nanvar(frac, axis=1, ddof=1)

    # Handle cells with all-NaN or zero variance gracefully
    bad = ~np.isfinite(m) | ~np.isfinite(v) | (v <= 0)
    # For such cells, fall back to a weak, symmetric prior m=0.5, v ~ 1/12
    m[bad] = 0.5
    v[bad] = (m[bad] * (1 - m[bad])) / 3.0

    # Clamp means away from 0 and 1
    m = np.clip(m, eps, 1.0 - eps)

    # Method-of-moments for Beta(alpha, beta)
    s = m * (1.0 - m) / (v + eps) - 1.0
    s = np.maximum(s, 1e-2)  # avoid negative or insanely small sums

    alpha = m * s
    beta = (1.0 - m) * s

    return alpha, beta, m





def beta_binomial_posterior_normalized(
    mc: Array,
    cov: Array,
    eps: float = 1e-8,
    normalize: bool = True,
):
    """
    Cell-wise Beta prior:
      - estimate alpha_i, beta_i per cell
      - posterior mean: post_ij = (alpha_i + mc_ij) / (alpha_i + beta_i + cov_ij)
      - if normalize=True: return post_ij / m_i, where m_i = alpha_i / (alpha_i + beta_i)
    """
    mc = mc.astype(float)
    cov = cov.astype(float)

    alpha, beta, m = estimate_beta_prior_per_cell(mc, cov, eps=eps)

    alpha_mat = alpha[:, None]
    beta_mat  = beta[:, None]
    m_mat     = m[:, None]

    denom = alpha_mat + beta_mat + cov
    post = (alpha_mat + mc) / (denom + eps)

    if normalize:
        out = post / (m_mat + eps)
    else:
        out = post

    return out, alpha, beta, m



def simulate_two_type_methylation(cfg: SimConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)

    # Cell-type labels: 0 and 1 with given asymmetry
    n1 = int(round(cfg.n_cells * cfg.celltype1_fraction))
    n1 = max(1, min(cfg.n_cells - 1, n1))  # ensure both types present
    n2 = cfg.n_cells - n1
    cell_type = np.concatenate(
        [np.zeros(n1, dtype=np.int8), np.ones(n2, dtype=np.int8)]
    )

    # Differential vs non-differential loci
    is_diff = rng.random(cfg.n_loci) < cfg.diff_fraction
    assert np.any(is_diff), "No differential loci were drawn; increase diff_fraction."

    # Baseline betas for type A at all loci (U-shaped Beta)
    base = rng.beta(cfg.beta_1, cfg.beta_2, size=cfg.n_loci)

    # Type B locus-wise betas using logit-scale normal shift at diff loci
    beta_b = base.copy()
    beta_b[is_diff] = _apply_diff_logit(base[is_diff], rng, cfg.diff_sigma)

    # Per-cell per-locus betas: assign type means, then optional per-cell jitter
    beta_by_type = np.stack([base, beta_b], axis=0)  # (2, n_loci)
    B_mean = beta_by_type[cell_type, :]              # (n_cells, n_loci)

    if cfg.cell_jitter_sd > 0:
        Z = logit_clip(B_mean)
        Z = Z + rng.normal(loc=0.0, scale=cfg.cell_jitter_sd, size=Z.shape)
        B_true = _sigmoid(Z)
        Z_true = Z
    else:
        B_true = B_mean.copy()
        Z_true = logit_clip(B_true)

    # ---------------------------------------------------------
    # Read depths: ZINB with per-cell log-normal library effects
    # ---------------------------------------------------------

        # 1. Per-cell library size multipliers L_i
    L = rng.lognormal(mean=cfg.liblog_mu, sigma=cfg.liblog_sd, size=cfg.n_cells)
    L = L[:, None]  # (n_cells, 1)

    # 2. Cell-specific NB means: mu_ij = L_i * nb_mean, expanded over loci
    mu_matrix = L * cfg.nb_mean * np.ones((1, cfg.n_loci))  # (n_cells, n_loci)

    # 3. Per-locus scaling to induce heavy-tailed bin depth variation
    locus_scale = rng.lognormal(
        mean=cfg.locus_log_mu,
        sigma=cfg.locus_log_sd,
        size=cfg.n_loci,
    )
    mu_matrix = mu_matrix * locus_scale[None, :]

    # 4. NB draw using cell- and locus-specific means
    cov = _nbinom_mean_size(
        rng,
        mean=mu_matrix,
        size=cfg.nb_size,
        shape=mu_matrix.shape,
    )

    # 5. Zero inflation
    if cfg.zero_frac > 0:
        mask = rng.random(size=cov.shape) < cfg.zero_frac
        cov[mask] = 0


    # ---------------------------------------------------------
    # Methylated counts given coverage and B_true
    # ---------------------------------------------------------
    # No mask needed: Binomial(n=0, p) is always 0, so this is safe.
    mc = rng.binomial(cov, B_true).astype(np.int32)

    return SimulationResult(
        B_true=B_true,
        Z_true=Z_true,
        cov=cov,
        mc=mc,
        cell_type=cell_type,
        is_diff=is_diff,
    )

import numpy as np

# --------- helpers ---------

def logit_clip(p, eps=1e-4):
    """Stable logit transform for values in [0,1]."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1.0 - p))

def pca_scores(X: Array, k: int) -> Array:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k]


def mean_explained_variance(U_true: Array, U_hat: Array) -> float:
    k = min(U_true.shape[1], U_hat.shape[1])
    U_true = U_true[:, :k]
    U_hat = U_hat[:, :k]
    proj = U_hat.T @ U_true
    norms = np.linalg.norm(proj, axis=0)
    return float(norms.mean())


def pairwise_euclidean_distances(X: Array) -> Array:
    """
    Return condensed vector of all pairwise Euclidean distances between rows of X
    (i.e. upper triangle of the full distance matrix, flattened).
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


def _pearson_corr(x: Array, y: Array) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0:
        return np.nan
    return float((x * y).sum() / denom)


def _rankdata_1d(x: Array) -> Array:
    """
    Simple rankdata implementation (average rank for ties).
    Returns 1-based ranks as floats.
    """
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        # find tie block [i, j)
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        # average 1-based rank for indices i..j-1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _spearman_corr(x: Array, y: Array) -> float:
    rx = _rankdata_1d(x)
    ry = _rankdata_1d(y)
    return _pearson_corr(rx, ry)


def distance_correlation(U_true: Array, U_hat: Array) -> dict:
    """
    Distance correlation between two embeddings U_true, U_hat (cells × d).

    Steps:
      1. compute all pairwise Euclidean distances between cells in each embedding;
      2. vectorize the upper-triangular distances;
      3. compute Pearson and Spearman correlation between these vectors.
    """
    d_true = pairwise_euclidean_distances(U_true)
    d_hat = pairwise_euclidean_distances(U_hat)

    mask = np.isfinite(d_true) & np.isfinite(d_hat)
    d_true = d_true[mask]
    d_hat = d_hat[mask]

    pearson = _pearson_corr(d_true, d_hat)
    spearman = _spearman_corr(d_true, d_hat)

    return {"pearson": pearson, "spearman": spearman}


def intra_inter_distance_stats(embedding: Array, labels: Array) -> dict:
    """
    For each cell type c:
      - mean intra-type distance (within c)
      - mean inter-type distance (between c and all other cells)
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

        # Intra-type distances
        if X_c.shape[0] < 2:
            intra = np.nan
        else:
            intra = float(pairwise_euclidean_distances(X_c).mean())

        # Inter-type distances
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

        stats[int(c)] = {
            "n_cells": int(X_c.shape[0]),
            "intra_mean": intra,
            "inter_mean": inter,
            "ratio_inter_over_intra": float(ratio) if np.isfinite(ratio) else np.nan,
        }

    return stats


def estimate_beta_prior_per_feature(
    mc: Array,
    cov: Array,
    min_nonzero: int = 50,
    global_alpha: float = 0.5,
    global_beta: float = 0.5,
    eps: float = 1e-8,
):
    """
    Estimate per-feature (per locus) Beta prior parameters alpha_j, beta_j
    using method-of-moments on cell-wise proportions at each locus.

    This mirrors the R function `estimate_beta_priors()` you pasted:
      - For each locus j, use cells with cov_ij > 0
      - p_hat_ij = mc_ij / cov_ij
      - If at least `min_nonzero` cells and finite variance -> estimate Beta(a_j,b_j)
      - Otherwise fall back to a global weak prior (global_alpha, global_beta)

    Returns:
        alpha_feat: (n_loci,) array
        beta_feat:  (n_loci,) array
        m_feat:     (n_loci,) prior mean alpha/(alpha+beta)
    """
    mc = mc.astype(float)
    cov = cov.astype(float)
    n_cells, n_loci = mc.shape

    alpha_feat = np.full(n_loci, global_alpha, dtype=float)
    beta_feat  = np.full(n_loci, global_beta, dtype=float)

    # Prior means for convenience
    m_feat = np.empty(n_loci, dtype=float)

    for j in range(n_loci):
        cov_j = cov[:, j]
        mc_j  = mc[:, j]

        nz = cov_j > 0
        if nz.sum() < min_nonzero:
            # Not enough non-zero cells: keep global prior
            a = global_alpha
            b = global_beta
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                p_hat = mc_j[nz] / cov_j[nz]

            # Remove NaN/inf just in case
            p_hat = p_hat[np.isfinite(p_hat)]
            if p_hat.size == 0:
                a = global_alpha
                b = global_beta
            else:
                m = float(p_hat.mean())
                v = float(p_hat.var(ddof=1))

                if v > 0 and 0 < m < 1:
                    # Method-of-moments for Beta(a,b)
                    s = m * (1.0 - m) / (v + eps) - 1.0
                    if np.isfinite(s) and s > 0:
                        a = m * s
                        b = (1.0 - m) * s
                    else:
                        a = global_alpha
                        b = global_beta
                else:
                    a = global_alpha
                    b = global_beta

        alpha_feat[j] = a
        beta_feat[j]  = b
        m_feat[j]     = a / (a + b) if (a + b) > 0 else 0.5

    return alpha_feat, beta_feat, m_feat



def beta_binomial_posterior_featurewise(
    mc: Array,
    cov: Array,
    min_nonzero: int = 50,
    global_alpha: float = 0.5,
    global_beta: float = 0.5,
    eps: float = 1e-8,
):
    """
    Compute posterior methylation levels using a per-feature (per-locus)
    Beta prior Beta(alpha_j, beta_j).

    For each locus j:
      alpha_j, beta_j = estimate_beta_prior_per_feature(...)
      post_ij = (mc_ij + alpha_j) / (cov_ij + alpha_j + beta_j)

    No per-cell normalization is applied here (unlike the allcools-style
    per-cell posterior normalization you already implemented).

    Returns:
        post:       (n_cells, n_loci) posterior mean betas
        alpha_feat: (n_loci,) alpha_j
        beta_feat:  (n_loci,) beta_j
        m_feat:     (n_loci,) prior means alpha_j / (alpha_j + beta_j)
    """
    mc = mc.astype(float)
    cov = cov.astype(float)

    alpha_feat, beta_feat, m_feat = estimate_beta_prior_per_feature(
        mc,
        cov,
        min_nonzero=min_nonzero,
        global_alpha=global_alpha,
        global_beta=global_beta,
        eps=eps,
    )

    # Broadcast priors across cells
    alpha_mat = alpha_feat.reshape(1, -1)  # (1, n_loci), will broadcast over rows
    beta_mat  = beta_feat.reshape(1, -1)

    denom = cov + alpha_mat + beta_mat
    post = (mc + alpha_mat) / (denom + eps)

    return post, alpha_feat, beta_feat, m_feat

def median_impute_betas(mc: Array, cov: Array, eps: float = 1e-8):
    """
    Median imputation baseline:
      - For loci where cov_ij > 0: beta_ij = mc_ij / cov_ij
      - For cov_ij = 0: impute using the median beta at that locus
      - If a locus has no covered cells: default median = 0.5

    Returns:
        B_imp: (n_cells, n_loci) matrix of imputed betas
    """
    mc = mc.astype(float)
    cov = cov.astype(float)
    n_cells, n_loci = mc.shape

    B = np.zeros_like(mc, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = mc / cov
    frac[cov <= 0] = np.nan  # missing

    B_imp = frac.copy()

    # Compute locus-wise medians
    medians = np.nanmedian(B_imp, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.5)  # fallback if all NaN

    # Impute missing positions
    missing_mask = ~np.isfinite(B_imp)
    B_imp[missing_mask] = np.broadcast_to(medians, B_imp.shape)[missing_mask]

    # Clamp just in case
    B_imp = np.clip(B_imp, eps, 1 - eps)

    return B_imp

def mean_impute_betas(mc: Array, cov: Array, eps: float = 1e-8):
    """
    Mean imputation baseline:
      - For loci where cov_ij > 0: beta_ij = mc_ij / cov_ij
      - For cov_ij = 0: impute using the *mean* beta at that locus
      - If a locus has no covered cells: default mean = 0.5

    Returns:
        B_imp: (n_cells, n_loci) matrix of imputed betas
    """
    mc = mc.astype(float)
    cov = cov.astype(float)
    n_cells, n_loci = mc.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = mc / cov
    frac[cov <= 0] = np.nan  # mark missing

    B_imp = frac.copy()

    # Compute locus-wise means
    means = np.nanmean(B_imp, axis=0)
    means = np.where(np.isfinite(means), means, 0.5)  # fallback if all NaN

    # Impute missing entries
    missing_mask = ~np.isfinite(B_imp)
    B_imp[missing_mask] = np.broadcast_to(means, B_imp.shape)[missing_mask]

    # Guard numerically
    B_imp = np.clip(B_imp, eps, 1 - eps)

    return B_imp