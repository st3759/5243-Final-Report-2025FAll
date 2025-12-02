import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

# ------------------------------
# 路径设置（和现有工程保持一致）
# ------------------------------
SP_PATH = Path("pancreas_work/adata_sp.h5ad")
SC_PATH = Path("pancreas_work/adata_sc.h5ad")
W_LINEAR_CSV = Path("pancreas_work/linear_proportions_lstsq.csv")

OUT_H5AD = Path("pancreas_work/adata_sp_card.h5ad")
OUT_CSV  = Path("pancreas_work/card_proportions.csv")

# ------------------------------
# 超参数（可以酌情调）
# ------------------------------
LAMBDA_REG = 0.1     # 空间正则 λ
LR         = 1e-4    # 学习率（小一点避免数值爆炸）
N_ITER     = 500     # 迭代次数
PRINT_EVERY = 50


# --------- 和 baseline_model 保持一致的一些小函数 ---------
def pick_layer(adata, prefer=("normalized", "log_normalized", "X")):
    """从 AnnData 中选择一个表达矩阵层，逻辑复制自 baseline_model.py。"""
    for key in prefer:
        if key == "X":
            return adata.X, "X"
        if hasattr(adata, "layers") and key in adata.layers:
            return adata.layers[key], key
    return adata.X, "X"


def to_dense(x):
    """把稀疏矩阵转成 dense numpy array。"""
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def assert_finite(name, arr):
    """检查矩阵里是否存在 NaN / Inf，有就直接报错，不做填补。"""
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        n_nan    = np.isnan(a).sum()
        n_posinf = np.isposinf(a).sum()
        n_neginf = np.isneginf(a).sum()
        raise ValueError(
            f"[ERR] {name} 包含非法值：nan={n_nan}, +inf={n_posinf}, -inf={n_neginf}"
        )
    print(f"[CHECK] {name}: all finite, shape={a.shape}")


# ------------------------------
# 1. 按照 baseline 的方式构造 Y, X_ref, W_init, L
# ------------------------------
def load_data():
    print("[INFO] Loading AnnData...")
    adata_sp = ad.read_h5ad(SP_PATH)
    adata_sc = ad.read_h5ad(SC_PATH)

    if "cell_type" not in adata_sc.obs:
        raise KeyError("adata_sc.obs 里缺少 'cell_type' 列（baseline 就依赖这个）")

    # ---- 1) 从 layer 里选出表达矩阵，和 baseline_model 完全同一逻辑 ----
    Y_mat, sp_layer = pick_layer(adata_sp)      # 可能是 genes×spots 或 spots×genes
    X_sc_all, sc_layer = pick_layer(adata_sc)   # 单细胞表达

    genes_sp = pd.Index(adata_sp.var_names)
    genes_sc = pd.Index(adata_sc.var_names)
    genes = genes_sp.intersection(genes_sc)
    if len(genes) < 500:
        raise ValueError(f"交集基因数太少：{len(genes)}")

    sp_idx = genes_sp.get_indexer(genes)
    sc_idx = genes_sc.get_indexer(genes)

    # 根据维度判断是 (cells×genes) 还是 (genes×cells)
    Y_sel = Y_mat[:, sp_idx] if Y_mat.shape[1] == len(genes_sp) else Y_mat[sp_idx, :]
    X_sc_sel = X_sc_all[:, sc_idx] if X_sc_all.shape[1] == len(genes_sc) else X_sc_all[sc_idx, :]

    Y = to_dense(Y_sel)      # 可能是 G×S 或 S×G
    X_sc = to_dense(X_sc_sel)

    # baseline 的做法：保证 Y 的“行”是 gene 轴
    if Y.shape[0] != len(genes) and Y.shape[1] == len(genes):
        Y = Y.T
    assert Y.shape[0] == len(genes), f"Y 基因轴错误：{Y.shape}"

    # ---- 2) 构造参考矩阵 X_ref（每个 cell type 的平均表达） ----
    cell_types = adata_sc.obs["cell_type"].astype(str).values
    ct_levels = pd.Index(sorted(pd.unique(cell_types)))

    X_cols = []
    for ct in ct_levels:
        mask = (cell_types == ct)
        if mask.sum() == 0:
            raise ValueError(f"[ERR] cell_type='{ct}' 在单细胞中没有细胞，无法求平均表达")
        X_cols.append(X_sc[mask].mean(axis=0))
    X_ref = np.column_stack(X_cols)   # 形状可能是 G×K 或 K×G

    # baseline 的做法：保证 X_ref 也是“基因为行”
    if X_ref.shape[0] != len(genes) and X_ref.shape[1] == len(genes):
        X_ref = X_ref.T
    assert X_ref.shape == (len(genes), len(ct_levels)), f"X_ref 形状异常：{X_ref.shape}"

    G, S = Y.shape
    K = X_ref.shape[1]
    print(f"[INFO] shapes: Genes={G}, Spots={S}, CellTypes={K}")

    # ---- 3) 读取 Laplacian L ----
    if "graph_L" not in adata_sp.obsp:
        raise KeyError("[ERR] adata_sp.obsp 里没有 'graph_L'，请先跑 mock_spacial.py")
    L = adata_sp.obsp["graph_L"]
    if not sparse.issparse(L):
        L = sparse.csr_matrix(L)
    L = L.astype(np.float64)

    if L.shape[0] != S or L.shape[1] != S:
        raise ValueError(f"[ERR] L 形状应为 (Spots×Spots) = ({S},{S})，实际是 {L.shape}")

    # ---- 4) baseline 的 W_init（线性最小二乘结果） ----
    if not W_LINEAR_CSV.exists():
        raise FileNotFoundError(f"[ERR] 找不到 {W_LINEAR_CSV}，请先跑 baseline_model.py")

    dfW_init = pd.read_csv(W_LINEAR_CSV, index_col=0)
    W_init = dfW_init.values.astype(np.float64)   # Spots×CellTypes = S×K
    ct_from_W = dfW_init.columns.astype(str)

    if W_init.shape != (S, K):
        raise ValueError(f"[ERR] W_init 形状异常：{W_init.shape}，应为 ({S},{K})")

    # 保证 cell type 顺序一致
    if list(ct_from_W) != list(ct_levels):
        # 用 baseline 的列顺序重排 X_ref
        order = [ct_levels.get_loc(ct) for ct in ct_from_W]
        X_ref = X_ref[:, order]
        ct_levels = ct_from_W
        print("[INFO] Re-ordered X_ref to match W_init columns")

    # ---- 5) 严格检查所有矩阵是否无 NaN / Inf ----
    assert_finite("Y (G×S)", Y)
    assert_finite("X_ref (G×K)", X_ref)
    assert_finite("W_init (S×K)", W_init)
    if sparse.issparse(L):
        assert_finite("L.data", L.data)
    else:
        assert_finite("L", L)

    return adata_sp, Y, X_ref, L, W_init, list(ct_levels)


# ------------------------------
# 2. 按论文目标函数优化 W
#     min_W ||Y - X_ref W^T||_F^2 + λ Tr(W^T L W)
# ------------------------------
def card_optimize(Y, X_ref, L, W_init,
                  lambda_reg=LAMBDA_REG,
                  lr=LR,
                  n_iter=N_ITER,
                  print_every=PRINT_EVERY):

    Y = np.asarray(Y, dtype=np.float64)          # G×S
    X = np.asarray(X_ref, dtype=np.float64)      # G×K
    W = np.asarray(W_init, dtype=np.float64)     # S×K

    G, S = Y.shape
    G2, K = X.shape
    assert G == G2
    assert W.shape == (S, K)

    if not sparse.issparse(L):
        L = sparse.csr_matrix(L)
    L = L.tocsr()

    print("[INFO] Start CARD optimization (full objective)...")
    for it in range(1, n_iter + 1):
        # ----- data term -----
        # recon = X_ref W^T   (G×K)(K×S) = G×S
        recon = X @ W.T
        E = recon - Y                     # G×S
        # grad_data: 2 * (E^T X)          (S×G)(G×K) = S×K
        grad_data = 2.0 * (E.T @ X)

        # ----- spatial term -----
        # Tr(W^T L W) 的梯度 = 2 L W
        LW = L.dot(W)                     # S×K
        grad_spatial = 2.0 * lambda_reg * LW

        grad = grad_data + grad_spatial

        # ---- 梯度下降更新 ----
        W -= lr * grad

        # ---- 投影：非负 + 每行和为 1 ----
        W[W < 0] = 0.0
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W /= row_sums

        if it % print_every == 0 or it == 1 or it == n_iter:
            recon = X @ W.T
            data_loss = np.linalg.norm(Y - recon, ord="fro") ** 2
            spatial_term = np.sum(W * L.dot(W))
            total_loss = data_loss + lambda_reg * spatial_term
            print(
                f"[iter {it:4d}] total={total_loss:.6e} | "
                f"data={data_loss:.6e} | spatial={spatial_term:.6e}"
            )

            # 再次检查没有 NaN（防止数值爆炸）
            if not np.isfinite(total_loss):
                raise FloatingPointError(
                    f"Loss 在 iter={it} 变成 NaN/Inf，"
                    f"请尝试减小 LR 或 LAMBDA_REG。"
                )

    return W


# ------------------------------
# 3. 主函数
# ------------------------------
def main():
    adata_sp, Y, X_ref, L, W_init, ct_names = load_data()

    W_card = card_optimize(Y, X_ref, L, W_init)

    # 写回 AnnData & CSV
    adata_sp.obsm["W_card"] = W_card
    adata_sp.uns["card_cell_types"] = ct_names

    dfW_card = pd.DataFrame(W_card, index=adata_sp.obs_names, columns=ct_names)
    dfW_card.to_csv(OUT_CSV)
    adata_sp.write_h5ad(OUT_H5AD, compression="lzf")

    print("[OK] Saved CARD proportions to:", OUT_CSV)
    print("[OK] Saved AnnData with W_card to:", OUT_H5AD)


if __name__ == "__main__":
    main()
