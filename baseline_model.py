

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import matplotlib.pyplot as plt


SP_PATH = Path("pancreas_work/adata_sp.h5ad")
SC_PATH = Path("pancreas_work/adata_sc.h5ad")
OUTDIR  = Path("pancreas_work")
PLOTDIR = OUTDIR / "plots_linear"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)


def pick_layer(adata, prefer=("normalized", "log_normalized", "X")):
    for key in prefer:
        if key == "X":
            return adata.X, "X"
        if hasattr(adata, "layers") and key in adata.layers:
            return adata.layers[key], key
    return adata.X, "X"

def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


adata_sp = ad.read_h5ad(SP_PATH)
adata_sc = ad.read_h5ad(SC_PATH)
assert "cell_type" in adata_sc.obs, "adata_sc.obs['cell_type'] 缺失"


Y_mat, sp_layer = pick_layer(adata_sp)           # G × S（或 S × G；下面会校正）
X_sc_all, sc_layer = pick_layer(adata_sc)

genes_sp = pd.Index(adata_sp.var_names)
genes_sc = pd.Index(adata_sc.var_names)
genes = genes_sp.intersection(genes_sc)
if len(genes) < 500:
    raise ValueError(f"交集基因过少：{len(genes)}")

sp_idx = genes_sp.get_indexer(genes)
sc_idx = genes_sc.get_indexer(genes)

Y_sel = Y_mat[:, sp_idx] if Y_mat.shape[1] == len(genes_sp) else Y_mat[sp_idx, :]
X_sc_sel = X_sc_all[:, sc_idx] if X_sc_all.shape[1] == len(genes_sc) else X_sc_all[sc_idx, :]

Y = to_dense(Y_sel)
X_sc = to_dense(X_sc_sel)


if Y.shape[0] != len(genes) and Y.shape[1] == len(genes):
    Y = Y.T
assert Y.shape[0] == len(genes), f"Y 基因轴错误：{Y.shape}"


cell_types = adata_sc.obs["cell_type"].astype(str).values
ct_levels = pd.Index(sorted(pd.unique(cell_types)))

X_cols = []
for ct in ct_levels:
    mask = (cell_types == ct)
    X_cols.append(X_sc[mask].mean(axis=0))
X_ref = np.column_stack(X_cols)


if X_ref.shape[0] != len(genes) and X_ref.shape[1] == len(genes):
    X_ref = X_ref.T
assert X_ref.shape == (len(genes), len(ct_levels)), f"X_ref 形状异常：{X_ref.shape}"

G, S = Y.shape
K = X_ref.shape[1]
print(f"[INFO] shapes: Genes={G}, Spots={S}, CellTypes={K}")

# ---------- lstsq ----------

W_hat = np.linalg.lstsq(X_ref, Y, rcond=None)[0].T

W_hat = np.clip(W_hat, 0, None)
row_sums = W_hat.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
W_hat = W_hat / row_sums

# ---------- write back to AnnData  ----------
adata_sp.obsm["W_linear"] = W_hat
adata_sp.uns["cell_types"] = ct_levels.tolist()
adata_sp.write_h5ad(OUTDIR / "adata_sp_with_linear.h5ad", compression="lzf")

dfW = pd.DataFrame(W_hat, index=adata_sp.obs_names, columns=ct_levels)
dfW.to_csv(OUTDIR / "linear_proportions_lstsq.csv")
print("Estimated W shape (spots × cell types):", W_hat.shape)
print("[OK] saved proportions ->", OUTDIR / "linear_proportions_lstsq.csv")

# ---------- residuals ----------
Y_hat = (X_ref @ dfW.values.T)
res_l2 = np.sqrt(((Y - Y_hat) ** 2).sum(axis=0))
pd.Series(res_l2, index=adata_sp.obs_names, name="residual_l2").to_csv(OUTDIR / "linear_residual_l2_lstsq.csv")
print("[OK] saved residuals ->", OUTDIR / "linear_residual_l2_lstsq.csv")

# ================= visualization =================
# 1) heat map for each cell
coords = adata_sp.obsm.get("spatial", None)
if coords is None or coords.shape[1] < 2:
    print("[WARN] not found obsm['spatial']，skip")
else:
    K_plot = min(6, K)
    for i, ct in enumerate(ct_levels[:K_plot]):
        fig = plt.figure(figsize=(5.5, 5))
        sca = plt.scatter(coords[:, 0], coords[:, 1], c=W_hat[:, i], s=4)
        plt.title(f"Linear mixture proportion — {ct}")
        plt.xlabel("spatial-1"); plt.ylabel("spatial-2")
        plt.colorbar(sca, fraction=0.046, pad=0.04, label="proportion")
        plt.tight_layout()
        plt.savefig(PLOTDIR / f"linear_prop_{i+1}_{ct}.png", dpi=180)
        plt.close()
    print("[OK] saved proportion heatmaps to:", PLOTDIR)

# 2) residual plot
plt.figure(figsize=(6, 4))
plt.hist(res_l2, bins=50)
plt.title("Residual L2 per spot (Linear Lstsq)")
plt.xlabel("Residual L2"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOTDIR / "linear_residual_hist.png", dpi=180)
plt.close()
print("[OK] saved:", PLOTDIR / "linear_residual_hist.png")

# 3)（可选）各细胞类型全局比例条形图
global_means = dfW.mean(axis=0).sort_values(ascending=False)
plt.figure(figsize=(6.5, 4))
plt.bar(global_means.index.astype(str), global_means.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean proportion")
plt.title("Global mean proportions (Linear Lstsq)")
plt.tight_layout()
plt.savefig(PLOTDIR / "linear_global_mean_props.png", dpi=180)
plt.close()
print("[OK] saved:", PLOTDIR / "linear_global_mean_props.png")
