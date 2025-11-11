import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os, psutil
from pathlib import Path
import matplotlib.pyplot as plt


H5AD_PATH = "dataset.h5ad"
WORKDIR   = "pancreas_work"
os.makedirs(WORKDIR, exist_ok=True)


adata_b = ad.read_h5ad(H5AD_PATH, backed="r")
print(adata_b)
print("obs:", list(adata_b.obs.columns))
print("var:", list(adata_b.var.columns))
print("obsm:", list(adata_b.obsm.keys()))
print("layers:", list(adata_b.layers.keys()))
print("obsp:", list(adata_b.obsp.keys()))


adata = ad.read_h5ad(H5AD_PATH)



use_layer = "log_normalized"
assert use_layer in adata.layers, f"{use_layer} not found in layers!"
adata.X = adata.layers[use_layer]

if {"n_counts"}.issubset(adata.obs.columns):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

# HVG
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000, subset=False)



sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)


FULL_PATH = str(Path(WORKDIR) / "pancreas_full_pca50_neighbors.h5ad")
adata.write_h5ad(FULL_PATH, compression="lzf")
print("Saved:", FULL_PATH, "| shape:", adata.shape)




