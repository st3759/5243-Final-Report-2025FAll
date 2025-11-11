
import anndata as ad
import scanpy as sc
import numpy as np
from pathlib import Path
from scipy import sparse

INFILE = "dataset.h5ad"
OUTDIR = Path("pancreas_work")
OUT_SP = OUTDIR / "adata_sp.h5ad"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("[INFO] load", INFILE)
adata = ad.read_h5ad(INFILE)

# 1) make sure the pca exist
if "X_pca" not in adata.obsm:
    sc.pp.pca(adata, n_comps=50)

# 2) Ai_j
if "distances" not in adata.obsp or "connectivities" not in adata.obsp:
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=10, method="umap")

adata.obsp["knn_connectivities"] = adata.obsp.get("connectivities", adata.obsp["knn_connectivities"])
adata.obsp["knn_distances"] = adata.obsp.get("distances", adata.obsp["knn_distances"])

# 3)Umap and knn as the 2d pic:

if "X_umap" not in adata.obsm:
    try:
        sc.tl.umap(adata, min_dist=0.5)
    except Exception:
        sc.tl.draw_graph(adata)
        adata.obsm["X_umap"] = adata.obsm["X_draw_graph_fa"]

if "X_umap" in adata.obsm and adata.obsm["X_umap"].shape[1] >= 2:
    adata.obsm["spatial"] = adata.obsm["X_umap"][:, :2]
else:
    adata.obsm["spatial"] = adata.obsm["X_pca"][:, :2]

A = adata.obsp["knn_connectivities"].tocsr().astype(np.float64)
A.setdiag(0); A.eliminate_zeros()
deg = np.asarray(A.sum(1)).ravel()
D = sparse.diags(deg, format="csr")
L = (D - A).tocsr()
adata.obsp["graph_L"] = L

# 5) save as spacial data
adata.write_h5ad(OUT_SP, compression="lzf")
print("[OK] saved pseudo-spatial:", OUT_SP, "| shape:", adata.shape)
print("[HINT] adata.obsm['spatial'].shape =", adata.obsm['spatial'].shape)
print("[HINT] adata.obsp keys:", list(adata.obsp.keys()))
