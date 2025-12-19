# glm_pca.py

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np

Array = np.ndarray


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def compute_binomial_nll(
    mc: Array,
    cov: Array,
    eta: Array,
    mask: Optional[Array] = None,
    eps: float = 1e-12,
) -> float:
    """
    Mean negative log-likelihood for Binomial(mc | cov, p), using
    p = sigmoid(eta), optionally restricted to entries where mask==True.

    We drop the Binomial coefficient term since it does not depend
    on (U, V, alpha); we only need relative NLL across parameter values.
    """
    mc = mc.astype(float)
    cov = cov.astype(float)
    p = _sigmoid(eta)

    if mask is not None:
        mc = mc[mask]
        cov = cov[mask]
        p = p[mask]

    if mc.size == 0:
        return np.nan

    # Guard against log(0)
    p = np.clip(p, eps, 1.0 - eps)

    nll = -(
        mc * np.log(p) +
        (cov - mc) * np.log(1.0 - p)
    ).sum()

    # Return mean per-observation NLL
    return float(nll / mc.size)


def fit_binomial_glm_pca_earlystop(
    mc: Array,
    cov: Array,
    n_components: int = 3,
    max_iter: int = 5000,
    lr: float = 5e-3,
    l2_reg: float = 1e-3,
    val_frac: float = 0.1,
    patience: int = 20,
    beta1: float = 0.9,
    beta2: float = 0.99,
    adam_eps: float = 1e-8,
    print_every: int = 100,
    seed: Optional[int] = None,
) -> Tuple[Array, Array, Array, Dict[str, Any], Dict[str, Array]]:
    """
    Fit Binomial GLM-PCA with Adam and early stopping based on a held-out set
    of individual (cell, locus) entries.

    Returns
    -------
    U_best, V_best, alpha_best, history, masks

    history contains:
        - "iters":       iteration numbers where NLL was evaluated
        - "train_nll":   train NLL at those iterations
        - "val_nll":     val NLL at those iterations
        - "best_iter":   iteration with best val NLL
        - "best_val_nll": best val NLL
        - "U_snapshots": list of U matrices at those iterations
    """
    mc = np.asarray(mc, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n_cells, n_loci = mc.shape

    rng = np.random.default_rng(seed)

    # ----------------------------------------
    # 1) Build train/validation masks on entries with cov > 0
    # ----------------------------------------
    observed_mask = cov > 0
    if val_frac > 0.0:
        rand = rng.random(size=cov.shape)
        val_mask = (observed_mask) & (rand < val_frac)
        train_mask = observed_mask & (~val_mask)
    else:
        val_mask = np.zeros_like(observed_mask, dtype=bool)
        train_mask = observed_mask

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())

    print(
        f"[binomial_glm_pca] using {n_train} train entries and "
        f"{n_val} val entries (val_frac={val_frac:.2f})"
    )

    # ----------------------------------------
    # 2) Initialize parameters
    # ----------------------------------------
    k = n_components
    U = 0.01 * rng.standard_normal(size=(n_cells, k))
    V = 0.01 * rng.standard_normal(size=(n_loci, k))
    alpha = np.zeros(n_loci, dtype=float)

    # Adam state
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_a = np.zeros_like(alpha)
    v_a = np.zeros_like(alpha)

    # ----------------------------------------
    # 3) Training loop with early stopping
    # ----------------------------------------
    train_nll_hist = []
    val_nll_hist = []
    iters = []
    U_snapshots = []   # <--- NEW: store U at evaluation times

    best_val_nll = np.inf
    best_iter = 0
    best_params = (U.copy(), V.copy(), alpha.copy())
    no_improve = 0

    for t in range(1, max_iter + 1):
        # Forward pass
        eta = alpha[None, :] + U @ V.T

        # Gradient of NLL wrt eta on TRAIN entries only.
        p = _sigmoid(eta)

        mc_train = np.where(train_mask, mc, 0.0)
        cov_train = np.where(train_mask, cov, 0.0)
        grad_eta = cov_train * p - mc_train  # d/deta (-log L)

        # Backprop to parameters
        grad_alpha = grad_eta.sum(axis=0)           # (n_loci,)
        grad_U = grad_eta @ V                      # (n_cells, k)
        grad_V = grad_eta.T @ U                    # (n_loci, k)

        # L2 regularization
        grad_U += 2.0 * l2_reg * U
        grad_V += 2.0 * l2_reg * V

        # Adam updates
        b1t = 1.0 - beta1**t
        b2t = 1.0 - beta2**t

        # U
        m_U = beta1 * m_U + (1.0 - beta1) * grad_U
        v_U = beta2 * v_U + (1.0 - beta2) * (grad_U * grad_U)
        U_step = (m_U / b1t) / (np.sqrt(v_U / b2t) + adam_eps)
        U -= lr * U_step

        # V
        m_V = beta1 * m_V + (1.0 - beta1) * grad_V
        v_V = beta2 * v_V + (1.0 - beta2) * (grad_V * grad_V)
        V_step = (m_V / b1t) / (np.sqrt(v_V / b2t) + adam_eps)
        V -= lr * V_step

        # alpha
        m_a = beta1 * m_a + (1.0 - beta1) * grad_alpha
        v_a = beta2 * v_a + (1.0 - beta2) * (grad_alpha * grad_alpha)
        a_step = (m_a / b1t) / (np.sqrt(v_a / b2t) + adam_eps)
        alpha -= lr * a_step

        # Monitoring / early stopping
        if (t % print_every == 0) or (t == 1) or (t == max_iter):
            eta = alpha[None, :] + U @ V.T
            train_nll = compute_binomial_nll(mc, cov, eta, mask=train_mask)
            val_nll = compute_binomial_nll(mc, cov, eta, mask=val_mask) if n_val > 0 else np.nan

            train_nll_hist.append(train_nll)
            val_nll_hist.append(val_nll)
            iters.append(t)
            U_snapshots.append(U.copy())   # <--- store current U

            print(
                f"[binomial_glm_pca] iter={t:5d}, "
                f"train_nll={train_nll:.6f}, val_nll={val_nll:.6f}"
            )

            # Early stopping on val NLL
            if n_val > 0:
                if val_nll < best_val_nll - 1e-6:  # small tolerance
                    best_val_nll = val_nll
                    best_iter = t
                    best_params = (U.copy(), V.copy(), alpha.copy())
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(
                        f"[binomial_glm_pca] early stopping at iter={t}, "
                        f"best_iter={best_iter}, best_val_nll={best_val_nll:.6f}"
                    )
                    break

    # Use best parameters (by val NLL)
    U_best, V_best, alpha_best = best_params

    history: Dict[str, Any] = {
        "iters": np.array(iters, dtype=int),
        "train_nll": np.array(train_nll_hist, dtype=float),
        "val_nll": np.array(val_nll_hist, dtype=float),
        "best_iter": best_iter,
        "best_val_nll": best_val_nll,
        "U_snapshots": U_snapshots,  # <--- new
    }

    masks: Dict[str, Array] = {
        "train_mask": train_mask,
        "val_mask": val_mask,
    }

    return U_best, V_best, alpha_best, history, masks
