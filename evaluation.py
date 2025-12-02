"""
evaluation.py

对比多个空间解卷积模型（linear / Lasso / CARD / DestVI / RCTD）的效果。

主要做三类评价：
1) 表达重构质量：per-spot residual L2 & R²
2) 空间平滑度：Tr(W^T L W) / S（S 为 spot 数，越小越平滑）
3) 全局 cell-type 分布：各模型的 global mean proportion，对比 & 保存

运行前要求：
- 已有：
    pancreas_work/adata_sp.h5ad
    pancreas_work/adata_sc.h5ad
    （由 data_sclicing.py + mock_spacial.py 生成）
- baseline_model.py 已跑过，生成：
    pancreas_work/linear_proportions_lstsq.csv
    pancreas_work/lasso_proportions.csv
- 若存在：
    pancreas_work/card_proportions.csv
    pancreas_work/destvi_proportions.csv
    pancreas_work/rctd_proportions.csv
  则一并参与评价（不存在就自动跳过）
"""

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import matplotlib.pyplot as plt

# ----------------- 路径设置 -----------------
WORKDIR = Path("pancreas_work")
SP_PATH = WORKDIR / "adata_sp.h5ad"
SC_PATH = WORKDIR / "adata_sc.h5ad"

# 各模型的 proportion CSV（有就评，没有就跳过）
CSV_LINEAR = WORKDIR / "linear_proportions_lstsq.csv"
CSV_LASSO = WORKDIR / "lasso_proportions.csv"
CSV_CARD = WORKDIR / "opt_card_proportions.csv"
CSV_DESTVI = WORKDIR / "destvi_proportions.csv"
CSV_RCTD = WORKDIR / "rctd_proportions.csv"  # 预留给你后面写的 RCTD baseline

OUTDIR = WORKDIR / "evaluation"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ----------------- 一些小工具函数（和 baseline/CARD 对齐） -----------------
def pick_layer(adata, prefer=("normalized", "log_normalized", "X")):
    """从 AnnData 中选一个表达矩阵层，用法和 baseline_model / CARD 一致。"""
    for key in prefer:
        if key == "X":
            return adata.X, "X"
        if hasattr(adata, "layers") and key in adata.layers:
            return adata.layers[key], key
    return adata.X, "X"


def to_dense(x):
    """把稀疏矩阵变成 dense numpy array。"""
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def assert_finite(name, arr):
    """检查矩阵是否有 NaN / Inf，有就直接报错。"""
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        n_nan = np.isnan(a).sum()
        n_posinf = np.isposinf(a).sum()
        n_neginf = np.isneginf(a).sum()
        raise ValueError(
            f"[ERR] {name} 包含非法值：nan={n_nan}, +inf={n_posinf}, -inf={n_neginf}"
        )
    print(f"[CHECK] {name}: all finite, shape={a.shape}")


# ----------------- 1. 构造 Y, X_ref, L -----------------
def load_expression_and_graph():
    """
    从 adata_sp / adata_sc 里构造：
    - Y: G × S 空间表达矩阵（基因为行）
    - X_ref: G × K cell-type 参考表达
    - L: S × S 图 Laplacian（来自 mock_spacial.py）
    - ct_levels: cell-type 名称列表（canonical 顺序）
    - adata_sp: 基础的 spatial AnnData（用于 obs_names / L）
    """
    print("[INFO] Loading AnnData...")
    adata_sp = ad.read_h5ad(SP_PATH)
    adata_sc = ad.read_h5ad(SC_PATH)

    if "cell_type" not in adata_sc.obs:
        raise KeyError("adata_sc.obs 里必须有 'cell_type' 列（和 baseline 对齐）")

    # ---------- 选表达矩阵 ----------
    Y_mat, sp_layer = pick_layer(adata_sp)
    X_sc_all, sc_layer = pick_layer(adata_sc)

    genes_sp = pd.Index(adata_sp.var_names)
    genes_sc = pd.Index(adata_sc.var_names)
    genes = genes_sp.intersection(genes_sc)
    if len(genes) < 500:
        raise ValueError(f"交集基因太少：{len(genes)}")

    sp_idx = genes_sp.get_indexer(genes)
    sc_idx = genes_sc.get_indexer(genes)

    # 判断是 (cells/spot × genes) 还是 (genes × cells/spot)
    Y_sel = Y_mat[:, sp_idx] if Y_mat.shape[1] == len(genes_sp) else Y_mat[sp_idx, :]
    X_sc_sel = X_sc_all[:, sc_idx] if X_sc_all.shape[1] == len(genes_sc) else X_sc_all[sc_idx, :]

    Y = to_dense(Y_sel)
    X_sc = to_dense(X_sc_sel)

    # 保证 Y 行是基因轴
    if Y.shape[0] != len(genes) and Y.shape[1] == len(genes):
        Y = Y.T
    assert Y.shape[0] == len(genes), f"Y 基因轴错误：{Y.shape}"

    # ---------- 构造参考矩阵 X_ref（按 cell_type 求均值） ----------
    cell_types = adata_sc.obs["cell_type"].astype(str).values
    ct_levels = pd.Index(sorted(pd.unique(cell_types)))

    X_cols = []
    for ct in ct_levels:
        mask = (cell_types == ct)
        if mask.sum() == 0:
            raise ValueError(f"[ERR] cell_type='{ct}' 在单细胞中没有细胞")
        X_cols.append(X_sc[mask].mean(axis=0))
    X_ref = np.column_stack(X_cols)

    # 保证 X_ref 也是 G × K
    if X_ref.shape[0] != len(genes) and X_ref.shape[1] == len(genes):
        X_ref = X_ref.T
    assert X_ref.shape == (len(genes), len(ct_levels)), f"X_ref 形状异常：{X_ref.shape}"

    # ---------- Laplacian L ----------
    if "graph_L" not in adata_sp.obsp:
        raise KeyError("[ERR] adata_sp.obsp 里没有 'graph_L'，请先跑 mock_spacial.py")
    L = adata_sp.obsp["graph_L"]
    if not sparse.issparse(L):
        L = sparse.csr_matrix(L)
    L = L.astype(np.float64)

    G, S = Y.shape
    if L.shape != (S, S):
        raise ValueError(f"[ERR] L 形状应为 ({S},{S})，实际是 {L.shape}")

    # finite 检查
    assert_finite("Y (G×S)", Y)
    assert_finite("X_ref (G×K)", X_ref)
    if sparse.issparse(L):
        assert_finite("L.data", L.data)
    else:
        assert_finite("L", L)

    print(f"[INFO] shapes: Genes={G}, Spots={S}, CellTypes={len(ct_levels)}")

    return adata_sp, Y, X_ref, L, list(ct_levels)


# ----------------- 2. 读取各模型的 W，并对齐 cell-type 顺序 -----------------
def align_W_to_ct(dfW, canonical_cts):
    """
    把 dfW 的列对齐到 canonical_cts：
    - 没有的 cell-type 补 0
    - 多出来的 cell-type 直接丢弃
    返回：S × K numpy array
    """
    # 先保证 index 不乱
    dfW = dfW.copy()
    # 对齐列
    for ct in canonical_cts:
        if ct not in dfW.columns:
            dfW[ct] = 0.0
    # 只保留 canonical_cts，并按顺序重排
    dfW = dfW[canonical_cts]
    return dfW.values.astype(float)


def load_model_weights(adata_sp, ct_levels):
    """
    尝试加载多种模型的 W（spots × K）：
    - linear
    - lasso
    - card
    - destvi
    - rctd（如果你之后实现了的话）
    返回：models = {name: {"W": W, "ct_names": ct_levels}}
    """
    models = {}
    S = adata_sp.n_obs
    canonical_cts = list(ct_levels)

    # 定义一个小辅助
    def try_load(csv_path, name):
        if not csv_path.exists():
            print(f"[WARN] {name}: 找不到 {csv_path.name}，跳过")
            return None
        dfW = pd.read_csv(csv_path, index_col=0)
        if dfW.shape[0] != S:
            raise ValueError(
                f"[ERR] {name}: row 数 {dfW.shape[0]} != spot 数 {S}，"
                f"检查是否用的是同一个 adata_sp"
            )
        W = align_W_to_ct(dfW, canonical_cts)
        return W

    # 1) Linear
    W_linear = try_load(CSV_LINEAR, "Linear")
    if W_linear is not None:
        models["Linear"] = {"W": W_linear, "ct_names": canonical_cts}

    # 2) Lasso
    W_lasso = try_load(CSV_LASSO, "Lasso")
    if W_lasso is not None:
        models["Lasso"] = {"W": W_lasso, "ct_names": canonical_cts}

    # 3) CARD
    W_card = try_load(CSV_CARD, "opt_CARD")
    if W_card is not None:
        models["CARD"] = {"W": W_card, "ct_names": canonical_cts}

    # 4) DestVI
    if CSV_DESTVI.exists():
        dfW_dest = pd.read_csv(CSV_DESTVI, index_col=0)
        if dfW_dest.shape[0] != S:
            raise ValueError(
                f"[ERR] DestVI: row 数 {dfW_dest.shape[0]} != spot 数 {S}"
            )
        # DestVI 可能列名和 canonical_cts 一样，如果不一样就对齐
        W_dest = align_W_to_ct(dfW_dest, canonical_cts)
        models["DestVI"] = {"W": W_dest, "ct_names": canonical_cts}
    else:
        print("[WARN] DestVI: destvi_proportions.csv 不存在，跳过")

    # 5) RCTD（预留）
    if CSV_RCTD.exists():
        dfW_rctd = pd.read_csv(CSV_RCTD, index_col=0)
        if dfW_rctd.shape[0] != S:
            raise ValueError(
                f"[ERR] RCTD: row 数 {dfW_rctd.shape[0]} != spot 数 {S}"
            )
        W_rctd = align_W_to_ct(dfW_rctd, canonical_cts)
        models["RCTD"] = {"W": W_rctd, "ct_names": canonical_cts}
    else:
        print("[INFO] RCTD: 目前没有 rctd_proportions.csv，如后续添加会自动参与评估。")

    print("[INFO] loaded models:", ", ".join(models.keys()))
    if not models:
        raise RuntimeError("[ERR] 没有任何模型的 proportion 文件被找到，检查路径和前置脚本。")

    return models


# ----------------- 3. 计算指标：重构误差 & R² & 空间平滑度 -----------------
def evaluate_models(Y, X_ref, L, models):
    """
    对每个模型计算：
    - per-spot residual L2
    - per-spot R²
    - per-spot 空间 penalty（W * (L W) 按行和）
    并汇总一个 summary DataFrame。
    """
    G, S = Y.shape
    G2, K = X_ref.shape
    assert G == G2

    # 为了方便运算，将 L 转成 CSR
    if not sparse.issparse(L):
        L = sparse.csr_matrix(L)
    L = L.tocsr()

    # 用同一个总方差做 R²
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)  # shape (S,)

    per_model_metrics = []
    per_spot_metrics = []

    for name, info in models.items():
        print(f"[EVAL] {name} ...")
        W = info["W"]  # S × K
        if W.shape[0] != S:
            raise ValueError(f"[ERR] {name}: W 形状 {W.shape} 和 spot 数 {S} 不一致")

        # ----------- 重构 ----------
        # recon = X_ref W^T  => G × S
        Y_hat = X_ref @ W.T
        res = Y - Y_hat  # G × S

        # per-spot residual L2
        res_l2 = np.sqrt((res ** 2).sum(axis=0))  # 长度 S

        # per-spot R²
        ss_res = (res ** 2).sum(axis=0)
        R2 = 1.0 - ss_res / ss_tot

        # ----------- 空间平滑度 ----------
        # penalty_s = sum_k W[s, k] * (L W)[s, k]
        LW = L.dot(W)  # S × K
        spatial_per_spot = (W * LW).sum(axis=1)  # 长度 S
        total_spatial = spatial_per_spot.sum()

        # 聚合成 summary
        per_model_metrics.append({
            "model": name,
            "mean_res_l2": float(np.mean(res_l2)),
            "median_res_l2": float(np.median(res_l2)),
            "mean_R2": float(np.mean(R2)),
            "median_R2": float(np.median(R2)),
            "mean_spatial_penalty": float(np.mean(spatial_per_spot)),
            "total_spatial_penalty": float(total_spatial),
        })

        df_spot = pd.DataFrame({
            "model": name,
            "spot_idx": np.arange(S),
            "res_l2": res_l2,
            "R2": R2,
            "spatial_penalty": spatial_per_spot,
        })
        per_spot_metrics.append(df_spot)

    summary_df = pd.DataFrame(per_model_metrics).sort_values("mean_res_l2")
    per_spot_df = pd.concat(per_spot_metrics, axis=0, ignore_index=True)

    summary_df.to_csv(OUTDIR / "evaluation_summary_per_model.csv", index=False)
    per_spot_df.to_csv(OUTDIR / "evaluation_metrics_per_spot_long.csv", index=False)

    print("[OK] 保存汇总指标 ->", OUTDIR / "evaluation_summary_per_model.csv")
    print("[OK] 保存逐 spot 指标 ->", OUTDIR / "evaluation_metrics_per_spot_long.csv")

    return summary_df, per_spot_df


# ----------------- 4. 额外：全局 cell-type 分布 & 简单可视化 -----------------
def analyze_global_composition(models, ct_names):
    """统计每个模型的全局 cell-type mean proportion，方便肉眼对比。"""
    records = []
    for name, info in models.items():
        W = info["W"]  # S × K
        global_means = W.mean(axis=0)  # K
        for ct, m in zip(ct_names, global_means):
            records.append({
                "model": name,
                "cell_type": ct,
                "mean_prop": float(m)
            })
    df = pd.DataFrame(records)
    df.to_csv(OUTDIR / "global_celltype_composition.csv", index=False)
    print("[OK] 保存全局 cell-type 分布 ->", OUTDIR / "global_celltype_composition.csv")
    return df


def quick_plots(summary_df, per_spot_df):
    """画几张简单图，帮助写 report。"""
    # 1) 模型级 R² / residual 对比条形图
    plt.figure(figsize=(6, 4), dpi=200)
    plt.bar(summary_df["model"], summary_df["mean_R2"])
    plt.ylabel("Mean R²")
    plt.title("Model comparison: mean R² per spot")
    plt.tight_layout()
    plt.savefig(OUTDIR / "model_mean_R2_bar.png")
    plt.close()

    plt.figure(figsize=(6, 4), dpi=200)
    plt.bar(summary_df["model"], summary_df["mean_res_l2"])
    plt.ylabel("Mean residual L2")
    plt.title("Model comparison: mean residual L2 per spot")
    plt.tight_layout()
    plt.savefig(OUTDIR / "model_mean_residual_bar.png")
    plt.close()

    # 2) R² 分布直方图（多个模型叠加）
    plt.figure(figsize=(7, 4), dpi=200)
    for name in summary_df["model"]:
        vals = per_spot_df.loc[per_spot_df["model"] == name, "R2"].values
        plt.hist(vals, bins=40, alpha=0.4, label=name, density=True)
    plt.xlabel("R²")
    plt.ylabel("Density")
    plt.title("R² distribution per spot (all models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "R2_hist_all_models.png")
    plt.close()

    # 3) 空间 penalty 对比条形图
    plt.figure(figsize=(6, 4), dpi=200)
    plt.bar(summary_df["model"], summary_df["mean_spatial_penalty"])
    plt.ylabel("Mean spatial penalty")
    plt.title("Model comparison: spatial smoothness (lower is smoother)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "model_spatial_penalty_bar.png")
    plt.close()

    print("[OK] 保存 evaluation 图像到:", OUTDIR)


# ----------------- 5. main -----------------
def main():
    # 1) 表达 & 图
    adata_sp, Y, X_ref, L, ct_levels = load_expression_and_graph()

    # 2) 加载各模型 W
    models = load_model_weights(adata_sp, ct_levels)

    # 3) 评价：重构 & R² & 空间平滑
    summary_df, per_spot_df = evaluate_models(Y, X_ref, L, models)

    # 4) 全局 cell-type 分布
    comp_df = analyze_global_composition(models, ct_levels)

    # 5) 一些 quick plots
    quick_plots(summary_df, per_spot_df)

    print("\n===== Evaluation Done =====")
    print(summary_df)


if __name__ == "__main__":
    main()

