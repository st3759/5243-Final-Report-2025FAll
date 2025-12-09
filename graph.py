import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
from matplotlib import gridspec
from pathlib import Path

WORKDIR = Path("pancreas_work")
SP_PATH = WORKDIR / "adata_sp_opt_card.h5ad"
OUT_FIG = WORKDIR / "card_schematic_pretty.png"

plt.style.use("default")  # 保持干净的默认风格

# ---------- 小工具：画饼图 ----------

def draw_pie(ax, x, y, fracs, radius, colors, start_angle=90):
    fracs = np.asarray(fracs, float)
    fracs = np.clip(fracs, 0.0, None)
    total = fracs.sum()
    if total <= 0:
        return
    fracs = fracs / total

    theta = start_angle
    for f, c in zip(fracs, colors):
        if f <= 0:
            continue
        dtheta = 360 * f
        wedge = Wedge((x, y), radius, theta, theta + dtheta,
                      facecolor=c, edgecolor="white", linewidth=0.15)
        ax.add_patch(wedge)
        theta += dtheta

# ---------- 主函数 ----------

def main():
    print("[CARD schematic] Loading data...")
    adata_sp = ad.read_h5ad(SP_PATH)

    W = adata_sp.obsm["W_card"]                 # (S, K)
    ct_names = list(adata_sp.uns["card_cell_types"])
    coords = adata_sp.obsm["spatial"]           # (S, 2)

    S, K = W.shape
    print(f"[INFO] spots={S}, cell types={K}")

    # 颜色：tab10/tab20 前 K 个
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i / max(K, 1)) for i in range(K)]

    # figure 布局：上面一行两个大图，下面三张小图
    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1.0], hspace=0.4, wspace=0.35)

    # ---------- Panel A: W 子矩阵 heatmap ----------
    n_spots_show = min(35, S)
    spot_idx = np.linspace(0, S - 1, n_spots_show, dtype=int)
    W_sub = W[spot_idx, :]

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(W_sub, aspect="auto")
    ax1.set_xticks(np.arange(K))
    ax1.set_xticklabels(ct_names, rotation=75, ha="right", fontsize=6)
    ax1.set_yticks(np.arange(n_spots_show))
    ax1.set_yticklabels(spot_idx, fontsize=6)
    ax1.set_xlabel("Cell type", fontsize=8)
    ax1.set_ylabel("Spot index", fontsize=8)
    ax1.set_title("Estimated cell-type proportions (subset)", fontsize=9, pad=5)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    # ---------- Panel B: 空间上的饼图 ----------
    ax2 = fig.add_subplot(gs[0, 1:])
    x = coords[:, 0]
    y = coords[:, 1]

    # 背景点（灰色）
    ax2.scatter(x, y, s=4, color="lightgrey", alpha=0.4, linewidths=0)

    # 抽样一些点画饼图
    n_pies = min(150, S)
    rng = np.random.default_rng(0)
    chosen = rng.choice(S, size=n_pies, replace=False)

    xr = x.max() - x.min()
    yr = y.max() - y.min()
    radius = 0.012 * max(xr, yr)  # 半径相对整体坐标调节

    for s in chosen:
        draw_pie(ax2, x[s], y[s], W[s, :], radius, colors)

    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Per-spot cell-type mixture (CARD)", fontsize=9, pad=5)

    # 加一个 cell-type legend
    legend_patches = [Patch(facecolor=c, edgecolor="none", label=ct)
                      for ct, c in zip(ct_names, colors)]
    # legend 放到图外右上角
    ax2.legend(handles=legend_patches,
               bbox_to_anchor=(1.02, 1.0),
               loc="upper left",
               fontsize=6,
               title="Cell types",
               title_fontsize=7,
               frameon=False)

    # ---------- Panel CDE: 三个典型 cell type 的空间 map ----------
    # 你可以换成自己想重点展示的 cell type
    target_ct = ["alpha", "beta", "ductal"]
    for i, ct in enumerate(target_ct):
        if ct not in ct_names:
            continue
        j = ct_names.index(ct)
        ax = fig.add_subplot(gs[1, i])
        vals = W[:, j]

        sc = ax.scatter(x, y, c=vals, s=6, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ct} proportion", fontsize=8)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle("CARD: Estimated cell-type proportions and spatial maps", fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print("[OK] Saved pretty CARD schematic figure ->", OUT_FIG)

if __name__ == "__main__":
    main()
