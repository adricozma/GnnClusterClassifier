# Study 2 Trained GNN Weights Registry

All best model checkpoints (`.pth`) are saved both in their respective experiment directories and mirrored in `GNN-jed/trained_models/study2/weights/`.

| Checkpoint File | Architecture | Features | Best Epoch | Val Loss | Val Acc | Core FNR | Test Acc | Test F1 | Size | Notes / Experiment Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `best_gat_clean.pth` | **GAT (Official)** | 5D Clean | 18 | 0.1470 | **94.20%** | **6.49%** | **93.04%** | **0.9035** | 361.2 KB | Canonical 5D Clean GAT (`dropout=0.2`, 5 rot/ep) — `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` |
| `best_gat_clean_ep18_backup.pth` | **GAT (Backup)** | 5D Clean | 18 | 0.1470 | **94.20%** | **6.49%** | **93.04%** | **0.9035** | 361.2 KB | Identical backup copy of canonical Ep 18 checkpoint |
| `best_gat_clean_dropout04_ep35.pth` | **GAT (Stochastic `p=0.4`)** | 5D Clean | 35 | 0.2071 | **91.89%** | 9.22% | **91.43%** | 0.8807 | 361.2 KB | Archived over-regularized run (`dropout=0.4` on all layers incl. `conv3`, 1 rot/ep) |
| `best_gcn_clean.pth` | **GCN (Baseline)** | 5D Clean | 19 | 0.1725 | **93.16%** | 10.32% | **92.17%** | 0.8886 | 120.0 KB | Isotropic 3-layer Residual GCN baseline — `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` |
| `best_gat_noisy.pth` | **GAT (Noisy 5D)** | 5D Noisy | 12 | 0.4014 | **81.76%** | 19.52% | **84.91%** | 0.7879 | 361.2 KB | Gaia DR3 5D Noisy GAT (affected by $1/\varpi_{\mathrm{obs}}$ 5D $k$-NN corruption & `dropout=0.4`) |
