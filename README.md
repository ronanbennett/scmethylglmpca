# scmethylglmpca
GLM-PCA for single-cell methylation data

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate igvf
```

The main fitting code can be run as follows (updated: 12/18/2025):

```python
from glm_pca import fit_binomial_glm_pca_earlystop
# Define mc and cov as anndata objects, representing cell x feature matrices.
U_glm, V_glm, alpha_glm, history_glm, masks_glm = fit_binomial_glm_pca_earlystop(
    mc,
    cov,
    n_components=10,
    max_iter=2000,
    lr=5e-3,
    l2_reg=1e-3,
    val_frac=0.1,
    patience=20,
    beta1=0.9,
    beta2=0.99,
    adam_eps=1e-8,
    print_every=200,
    seed=0,
)
# U: (n_cells, n_components) cell latent scores / embeddings (rows = cells)
# V: (n_loci, n_components) locus latent loadings (rows = loci)
# alpha: (n_loci,) locus-specific intercepts (baseline methylation levels)
```
