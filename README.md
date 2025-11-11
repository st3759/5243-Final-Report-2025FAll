# Human Pancreas Spatial Transcriptomics Baseline Project

This repository contains a set of reproducible Python scripts for preprocessing, slicing, pseudo-spatial construction, baseline deconvolution modeling, and visualization of a **human pancreas** dataset using **Scanpy** and **AnnData**.  


---
## Data Source

```
https://openproblems.bio/datasets/openproblems_v1/pancreas
```

##  Project Structure

```
Human pancreas project/
│
├── dataset.h5ad                 # Raw AnnData file (combined scRNA + spatial)
├── data_sclicing.py             # Split dataset into spatial and scRNA subsets
├── data_observation.py          # Preprocess, normalize, PCA, and neighbor graph
├── mock_spacial.py              # Construct pseudo-spatial coordinates and Laplacian
├── baseline_model.py            # Linear least-squares baseline deconvolution
├── data_visualization.py        # UMAP, heatmaps, cell-type distributions
└── pancreas_work/               # Output directory for intermediate and final files
```

---

## 1. Environment Setup

### Requirements  
```bash
python>=3.10
scanpy
anndata
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

Install dependencies using:
```bash
pip install scanpy anndata numpy pandas matplotlib seaborn scipy scikit-learn
```

---

## 2. Workflow Overview 

| Step | Script | Purpose                                                         | Input | Output |
|------|---------|-----------------------------------------------------------------|--------|---------|
| 1.   | `data_observation.py` | Preprocess, filter, normalize, PCA, neighbors                   | `dataset.h5ad` | `pancreas_full_pca50_neighbors.h5ad` |
| 2.   | `data_visualization.py` | Visualize UMAP, heatmaps, residuals, and cell type distribution | processed `.h5ad` | `.png` plots |
| 3.   | `data_sclicing.py` | Identify scRNA data                                             | `dataset.h5ad` | `adata_sc.h5ad` |
| 4.   | `mock_spacial.py` | Construct spatial coordinates using KNN/UMAP                    | `dataset.h5ad` | `adata_sp.h5ad` (with `.obsm['spatial']`) |
| 5.   | `baseline_model.py` | Perform baseline model (linear least squares)                   | `adata_sp.h5ad`, `adata_sc.h5ad` | `linear_proportions_lstsq.csv`, `linear_residual_l2_lstsq.csv` |


---

## 3. Script Details

### `data_observation.py`
- **Function:** Basic QC and dimensionality reduction.
- **Key Operations:**
  - Load full dataset (`dataset.h5ad`)
  - Apply `log_normalized` layer
  - Filter low-quality cells/genes
  - Identify 2,000 highly variable genes
  - Scale → PCA (50 PCs) → Compute neighbors
- **Output:** `pancreas_work/pancreas_full_pca50_neighbors.h5ad`

---

### `data_visualization.py`
- **Function:** Generate UMAP & expression-level heatmaps.
- **Steps:**
  - Compute UMAP from PCA representation.
  - Save colored UMAP plots by cell type / batch / tech.
  - Generate bar plot for cell-type counts.
  - Build top-300-spot similarity heatmap using cosine similarity.
- **Output Files:**
  - `umap_by_celltype.png`
  - `celltype_bar.png`
  - `heatmap_per_spot.png`

---

### `data_sclicing.py`
- **Function:** Automatically detects and separates **spatial** vs **scRNA-seq** modalities.
- **Logic:**  
  - Checks `.obs["tech"]`, `in_tissue`, `array_row`, etc.  
  - Assigns boolean masks to extract subsets.  
  - Writes `adata_sp.h5ad` and `adata_sc.h5ad`.

---

### `mock_spacial.py`
- **Function:** Build a **spatial** representation from scRNA data.
- **Pipeline:**
  - Run PCA & KNN to get adjacency matrices (`connectivities`, `distances`)
  - Derive UMAP or fallback to PCA for `.obsm["spatial"]`
  - Save spatial data for later modeling

---

### `baseline_model.py`
- **Function:** Implement linear least-squares (Lstsq) baseline deconvolution.
- **Workflow:**
  1. Align genes between `adata_sp` and `adata_sc`.
  2. Compute reference expression matrix `X_ref` by averaging each cell type.
  3. Estimate spot-celltype proportions `W_hat` via linear least squares.
  4. Normalize proportions per spot.
  5. Save `linear_proportions_lstsq.csv` & residual diagnostics.
- **Visualization Outputs:**
  - Per-cell-type heatmaps (`linear_prop_*.png`)
  - Residual histogram (`linear_residual_hist.png`)
  - Global mean bar chart (`linear_global_mean_props.png`)

---


## 4. Execution Order 

```bash
python data_observation.py
python data_visualization.py
python data_sclicing.py
python mock_spacial.py
python baseline_model.py
```

---

## 5. Output Summary

| Output Type | File | Description |
|--------------|------|-------------|
| Processed data | `pancreas_work/adata_sp_with_linear.h5ad` | spatial data with linear deconvolution results |
| CSV | `linear_proportions_lstsq.csv` | cell-type proportions per spot |
| CSV | `linear_residual_l2_lstsq.csv` | residual L2 errors |
| Figures | `linear_prop_*.png`, `linear_residual_hist.png`, `linear_global_mean_props.png` | model visualization |
| Figures | `umap_by_*.png`, `heatmap_per_spot.png` | UMAP & similarity maps |

---

## 6. Notes

- All scripts assume a single input file `dataset.h5ad`.
- Intermediate files are saved under `pancreas_work/`.
- The baseline linear model can later be replaced with **Cell2location**, **DestVI**, or **RCTD** for benchmarking.
- Ensure sufficient memory (>8 GB RAM) for PCA and matrix operations.

---
