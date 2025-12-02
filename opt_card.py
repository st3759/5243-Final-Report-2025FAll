import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import torch

# ------------------------------
# 路径设置（和现有工程保持一致）
# ------------------------------
SP_PATH = Path("pancreas_work/adata_sp.h5ad")
SC_PATH = Path("pancreas_work/adata_sc.h5ad")
W_LINEAR_CSV = Path("pancreas_work/linear_proportions_lstsq.csv")

OUT_H5AD = Path("pancreas_work/adata_sp_opt_card.h5ad")
OUT_CSV  = Path("pancreas_work/opt_card_proportions.csv")

# ------------------------------
# 超参数（可再调）
# ------------------------------
LAMBDA_REG  = 1.0      # 空间正则 λ（比之前 0.1 稍大，让平滑项更有存在感）
LR           = 1e-2     # Adam 学习率
N_ITER       = 400      # 迭代次数
PRINT_EVERY  = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Use device: {device}")


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
    """检查矩阵里是否存在 NaN / Inf，有就直接报错。"""
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
# 1. 按 baseline 的方式构造 Y, X_ref, W_init, L
# ------------------------------
def load_data():
    print("[INFO] Loading AnnData...")
    adata_sp = ad.read_h5ad(SP_PATH)
    adata_sc = ad.read_h5ad(SC_PATH)

    if "cell_type" not in adata_sc.obs:
        raise KeyError("adata_sc.obs 里缺少 'cell_type' 列（baseline 就依赖这个）")

    # ---- 1) 表达矩阵 ----
    Y_mat, sp_layer   = pick_layer(adata_sp)
    X_sc_all, sc_layer = pick_layer(adata_sc)

    genes_sp = pd.Index(adata_sp.var_names)
    genes_sc = pd.Index(adata_sc.var_names)
    genes = genes_sp.intersection(genes_sc)
    if len(genes) < 500:
        raise ValueError(f"交集基因数太少：{len(genes)}")

    sp_idx = genes_sp.get_indexer(genes)
    sc_idx = genes_sc.get_indexer(genes)

    Y_sel   = Y_mat[:, sp_idx]   if Y_mat.shape[1] == len(genes_sp) else Y_mat[sp_idx, :]
    X_sc_sel = X_sc_all[:, sc_idx] if X_sc_all.shape[1] == len(genes_sc) else X_sc_all[sc_idx, :]

    Y = to_dense(Y_sel)
    X_sc = to_dense(X_sc_sel)

    if Y.shape[0] != len(genes) and Y.shape[1] == len(genes):
        Y = Y.T
    assert Y.shape[0] == len(genes), f"Y 基因轴错误：{Y.shape}"

    # ---- 2) cell-type mean 表达 X_ref ----
    cell_types = adata_sc.obs["cell_type"].astype(str).values
    ct_levels = pd.Index(sorted(pd.unique(cell_types)))

    X_cols = []
    for ct in ct_levels:
        mask = (cell_types == ct)
        if mask.sum() == 0:
            raise ValueError(f"[ERR] cell_type='{ct}' 在单细胞中没有细胞")
        X_cols.append(X_sc[mask].mean(axis=0))
    X_ref = np.column_stack(X_cols)

    if X_ref.shape[0] != len(genes) and X_ref.shape[1] == len(genes):
        X_ref = X_ref.T
    assert X_ref.shape == (len(genes), len(ct_levels)), f"X_ref 形状异常：{X_ref.shape}"

    G, S = Y.shape
    K    = X_ref.shape[1]
    print(f"[INFO] shapes: Genes={G}, Spots={S}, CellTypes={K}")

    # ---- 3) Laplacian L ----
    if "graph_L" not in adata_sp.obsp:
        raise KeyError("[ERR] adata_sp.obsp 里没有 'graph_L'，请先跑 mock_spacial.py")
    L = adata_sp.obsp["graph_L"]
    if not sparse.issparse(L):
        L = sparse.csr_matrix(L)
    L = L.astype(np.float64)
    if L.shape != (S, S):
        raise ValueError(f"[ERR] L 形状应为 ({S},{S})，实际是 {L.shape}")

    # ---- 4) baseline linear 的 W_init ----
    if not W_LINEAR_CSV.exists():
        raise FileNotFoundError(f"[ERR] 找不到 {W_LINEAR_CSV}，请先跑 baseline_model.py")

    dfW_init = pd.read_csv(W_LINEAR_CSV, index_col=0)
    W_init = dfW_init.values.astype(np.float64)   # S×K
    ct_from_W = dfW_init.columns.astype(str)

    if W_init.shape != (S, K):
        raise ValueError(f"[ERR] W_init 形状异常：{W_init.shape}，应为 ({S},{K})")

    if list(ct_from_W) != list(ct_levels):
        order = [ct_levels.get_loc(ct) for ct in ct_from_W]
        X_ref = X_ref[:, order]
        ct_levels = ct_from_W
        print("[INFO] Re-ordered X_ref to match W_init columns")

    # ---- 5) 最后检查 ----
    assert_finite("Y (G×S)", Y)
    assert_finite("X_ref (G×K)", X_ref)
    assert_finite("W_init (S×K)", W_init)
    if sparse.issparse(L):
        assert_finite("L.data", L.data)

    return adata_sp, Y, X_ref, L, W_init, list(ct_levels)


# ------------------------------
# 2. 用 PyTorch + Adam 优化 CARD 目标
# ------------------------------
def scipy_L_to_torch_sparse(L_csr, device):
    """把 scipy csr_matrix 转成 torch.sparse_coo_tensor."""
    L_coo = L_csr.tocoo()
    indices = np.vstack((L_coo.row, L_coo.col))
    indices = torch.from_numpy(indices).long()
    values  = torch.from_numpy(L_coo.data.astype(np.float32))
    shape   = L_coo.shape
    L_torch = torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device)
    return L_torch.coalesce()


def card_optimize_torch(Y_np, X_np, L_csr, W_init_np,
                        lambda_reg=LAMBDA_REG,
                        lr=LR,
                        n_iter=N_ITER,
                        print_every=PRINT_EVERY):

    G, S = Y_np.shape
    G2, K = X_np.shape
    assert G == G2
    assert W_init_np.shape == (S, K)

    # 转成 torch，放到 device 上
    Y = torch.from_numpy(Y_np.astype(np.float32)).to(device)      # G×S
    X = torch.from_numpy(X_np.astype(np.float32)).to(device)      # G×K
    W = torch.nn.Parameter(torch.from_numpy(W_init_np.astype(np.float32)).to(device))  # S×K

    L = scipy_L_to_torch_sparse(L_csr, device)

    optimizer = torch.optim.Adam([W], lr=lr)

    print("[INFO] Start CARD optimization with PyTorch ...")
    for it in range(1, n_iter + 1):
        optimizer.zero_grad()

        # data term: ||Y - X W^T||_F^2
        recon = torch.matmul(X, W.t())              # G×S
        E = recon - Y
        data_loss = torch.sum(E * E)

        # spatial term: Tr(W^T L W) = sum_s,k W_sk * (L W)_sk
        LW = torch.sparse.mm(L, W)                  # S×K
        spatial_term = torch.sum(W * LW)

        loss = data_loss + lambda_reg * spatial_term
        loss.backward()

        optimizer.step()

        # 约束：W >= 0, 每行和为 1
        with torch.no_grad():
            W.clamp_(min=0.0)
            row_sums = W.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            W.div_(row_sums)

        if it % print_every == 0 or it == 1 or it == n_iter:
            dl = data_loss.detach().cpu().item()
            sl = spatial_term.detach().cpu().item()
            tl = loss.detach().cpu().item()
            print(f"[iter {it:4d}] total={tl:.4e} | data={dl:.4e} | spatial={sl:.4e}")

            if not np.isfinite(tl):
                raise FloatingPointError(
                    f"Loss 在 iter={it} 变成 NaN/Inf，"
                    f"请尝试减小学习率或 lambda_reg。"
                )

    W_final = W.detach().cpu().numpy().astype(np.float64)
    return W_final


# ------------------------------
# 3. 主函数
# ------------------------------
def main():
    adata_sp, Y, X_ref, L, W_init, ct_names = load_data()

    W_card = card_optimize_torch(Y, X_ref, L, W_init)

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
