import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import pandas as pd
from pathlib import Path


WORKDIR = "pancreas_work"
Path(WORKDIR).mkdir(exist_ok=True)

adata = ad.read_h5ad("pancreas_work/pancreas_full_pca50_neighbors.h5ad")

# ========= Step 1:  UMAP =========
sc.tl.umap(adata, min_dist=0.5)
color_cols = [c for c in ["cell_type", "celltype", "batch", "tech"] if c in adata.obs.columns]

for c in color_cols:
    sc.pl.umap(adata, color=c, frameon=False, title=f"UMAP colored by {c}", show=False)
    plt.savefig(Path(WORKDIR) / f"umap_by_{c}.png", dpi=180, bbox_inches="tight")
plt.close("all")

# ========= Step 2:cell type hist =========
ct_col = None
for cand in ["cell_type", "celltype"]:
    if cand in adata.obs.columns:
        ct_col = cand
        break

if ct_col:
    plt.figure(figsize=(9, 3))
    adata.obs[ct_col].value_counts().sort_values(ascending=False).plot(kind="bar")
    plt.ylabel("Cell count")
    plt.title("Cell-type distribution")
    plt.tight_layout()
    plt.savefig(Path(WORKDIR) / "celltype_bar.png", dpi=180)
    plt.close()

# ========= Step 3: heat map per spot =========
if ct_col:
    print("Building per-spot heatmap (this may take a minute)...")


    expr_by_ct = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                              index=adata.obs_names,
                              columns=adata.var_names)
    expr_by_ct["cell_type"] = adata.obs[ct_col].values


    ct_means = expr_by_ct.groupby("cell_type").mean()


    from sklearn.metrics.pairwise import cosine_similarity
    spot_to_ct = cosine_similarity(expr_by_ct.drop(columns=["cell_type"]), ct_means)
    heatmap_df = pd.DataFrame(spot_to_ct, index=adata.obs_names, columns=ct_means.index)


    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df.iloc[:300, :], cmap="viridis", cbar_kws={'label': 'Similarity'})
    plt.title("Per-spot similarity to each cell type (Top 300 spots shown)")
    plt.xlabel("Cell type")
    plt.ylabel("Spot index")
    plt.tight_layout()
    plt.savefig(Path(WORKDIR) / "heatmap_per_spot.png", dpi=180)
    plt.close()

print(" Figures saved in:", WORKDIR)
