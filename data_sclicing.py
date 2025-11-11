

import anndata as ad
import pandas as pd
from pathlib import Path


INFILE = "dataset.h5ad"
OUTDIR = "pancreas_work"


def guess_modalities(adata):
    obs = adata.obs
    cols = obs.columns

    is_spatial = pd.Series(False, index=obs.index)
    is_scrna = pd.Series(False, index=obs.index)


    if "tech" in cols:
        vals = obs["tech"].astype(str).str.lower().fillna("")
        is_spatial |= vals.str.contains("visium|spatial|slide", regex=True)
        is_scrna |= vals.str.contains("sc|single|rna|smart", regex=True)


    if "in_tissue" in cols or "array_row" in cols or "array_col" in cols or "spatial" in adata.obsm_keys():
        spatial_hint = pd.Series(False, index=obs.index)
        for c in ["in_tissue", "array_row", "array_col"]:
            if c in cols:
                spatial_hint |= obs[c].notnull()
        is_spatial |= spatial_hint


    if "cell_type" in cols:
        unassigned = ~(is_spatial | is_scrna)
        is_scrna |= unassigned

    # avoiding duplication
    both = is_spatial & is_scrna
    if both.any():
        is_scrna[both] = False

    return is_spatial.values, is_scrna.values


def main():
    inp = Path(INFILE)
    out = Path(OUTDIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {inp}")
    adata = ad.read_h5ad(str(inp))

    is_spatial, is_scrna = guess_modalities(adata)
    n_sp, n_sc = int(is_spatial.sum()), int(is_scrna.sum())
    print(f"[INFO] total={adata.n_obs} | spatial={n_sp} | scRNA={n_sc}")

    if n_sp > 0:
        adata_sp = adata[is_spatial].copy()
        adata_sp.write_h5ad(out / "adata_sp.h5ad", compression="lzf")
        print(f"[OK] Saved: {out/'adata_sp.h5ad'} | shape={adata_sp.shape}")
    else:
        print("[WARN] no spacial sample")

    if n_sc > 0:
        adata_sc = adata[is_scrna].copy()
        adata_sc.write_h5ad(out / "adata_sc.h5ad", compression="lzf")
        print(f"[OK] Saved: {out/'adata_sc.h5ad'} | shape={adata_sc.shape}")
    else:
        print("[WARN] no scRNA sample")

if __name__ == "__main__":
    main()
