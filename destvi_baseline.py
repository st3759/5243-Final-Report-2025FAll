# baseline_destvi.py
#
# 运行前：
#   pip install scvi-tools
#   adata_sc.obs['cell_type'] 存在

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import scvi

WORKDIR = Path("pancreas_work")
SP_PATH = WORKDIR / "adata_sp.h5ad"
SC_PATH = WORKDIR / "adata_sc.h5ad"

OUT_H5AD = WORKDIR / "adata_sp_destvi.h5ad"
OUT_CSV  = WORKDIR / "destvi_proportions.csv"


def main():
    print("[DestVI] Loading data...")
    adata_sp = ad.read_h5ad(SP_PATH)
    adata_sc = ad.read_h5ad(SC_PATH)

    if "cell_type" not in adata_sc.obs:
        raise KeyError("adata_sc.obs 里必须有 'cell_type' 列")

    # ---------- 基因对齐 ----------
    genes_sp = pd.Index(adata_sp.var_names)
    genes_sc = pd.Index(adata_sc.var_names)
    genes = genes_sp.intersection(genes_sc)

    if len(genes) < 500:
        raise ValueError(f"共同基因太少: {len(genes)}")

    adata_sp = adata_sp[:, genes].copy()
    adata_sc = adata_sc[:, genes].copy()

    # 使用 counts 作为 raw counts
    if "counts" in adata_sc.layers:
        adata_sc.X = adata_sc.layers["counts"].copy()
    if "counts" in adata_sp.layers:
        adata_sp.X = adata_sp.layers["counts"].copy()

    adata_sc.X = sparse.csr_matrix(adata_sc.X)
    adata_sp.X = sparse.csr_matrix(adata_sp.X)

    # ========= 1) 先在 single-cell 上训练 CondSCVI =========
    print("[DestVI] Training CondSCVI on scRNA reference...")
    scvi.model.CondSCVI.setup_anndata(
        adata_sc,
        layer=None,           # counts 在 X，如果你是放在 adata_sc.layers["counts"]，这里就写 layer="counts"
        labels_key="cell_type",
        # 如果有 batch 信息，可以加上，比如 batch_key="batch" 或 "tech"
        # batch_key="batch",
    )
    sc_model = scvi.model.CondSCVI(adata_sc, weight_obs=False)
    sc_model.train(max_epochs=120)

    # ========= 2) 在 spatial 上训练 DestVI =========
    print("[DestVI] Training DestVI on spatial data...")
    scvi.model.DestVI.setup_anndata(
        adata_sp,
        layer=None,           # 同理，如果 counts 在 layer 里就写对应名字
    )

    destvi = scvi.model.DestVI.from_rna_model(
        adata_sp,
        sc_model,             # 这里现在是 CondSCVI 模型
        # l1_reg=0.0,         # 需要更 sparse 的 proportion 可以调大，比如 50
    )
    destvi.train(max_epochs=250)

    # ========= 3) 提取 cell type proportions =========
    # get_proportions 返回 shape = n_spots × n_labels
    props = destvi.get_proportions()
    # props 是 numpy array 或 DataFrame
    if isinstance(props, pd.DataFrame):
        W_dest = props.values
        ct_names = props.columns.astype(str)
    else:
        W_dest = np.asarray(props, dtype=np.float64)
        ct_names = np.unique(adata_sc.obs["cell_type"].astype(str))

    # 行归一化成比例
    row_sums = W_dest.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_dest = W_dest / row_sums

    print("[DestVI] W shape:", W_dest.shape)

    adata_sp.obsm["W_destvi"] = W_dest
    adata_sp.uns["destvi_cell_types"] = list(ct_names)

    dfW_dest = pd.DataFrame(W_dest, index=adata_sp.obs_names, columns=ct_names)
    dfW_dest.to_csv(OUT_CSV)
    adata_sp.write_h5ad(OUT_H5AD, compression="lzf")

    print("[DestVI] Saved proportions to:", OUT_CSV)
    print("[DestVI] Saved AnnData to:", OUT_H5AD)


if __name__ == "__main__":
    main()
