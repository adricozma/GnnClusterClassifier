# Study 2 Trained GNN Weights Registry

All best model checkpoints (`.pth`) are saved both in their respective experiment directories and mirrored in `GNN-jed/trained_models/study2/weights/`.

| Checkpoint File | Architecture | Features | Best Epoch | Val Loss | Val Acc | Core FNR | Test Acc | Test F1 | Size | Experiment Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `best_gat_4obs_no_RA_clean.pth` | **GAT** | 4D | 1 | 0.2692 | **88.90%** | 8.41% | **Pending** | - | 357.9 KB | `GNN-jed/trained_models/study2/4obs_no_RA_clean/best_gat_4obs_no_RA_clean.pth` |
| `best_gat_clean.pth` | **GAT (Canonical 5D Clean, dropout=0.2)** | 5D | 18 | 0.1470 | **94.20%** | 6.49% | **93.04%** | 0.9035 | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` |
| `best_gat_clean_dropout04_ep35.pth` | **GAT (Stochastic 1 rot/ep, dropout=0.4)** | 5D | 35 | 0.2071 | **91.89%** | 9.22% | **91.43%** | 0.8807 | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean_dropout04_ep35.pth` |
| `best_gat_clean_ep18_backup.pth` | **GAT (5 rotations/epoch, dropout=0.2)** | 5D | 18 | 0.1470 | **94.20%** | 6.49% | **93.04%** | 0.9035 | 361.2 KB | `GNN-jed/trained_models/study2/weights/best_gat_clean_ep18_backup.pth` |
| `best_gat_noisy.pth` | **GAT** | 5D | 12 | 0.4014 | **81.76%** | 19.52% | **84.91%** | 0.7879 | 361.2 KB | `GNN-jed/trained_models/study2/noisy_5obs/best_gat_noisy.pth` |
| `best_gcn_clean.pth` | **GCN** | 5D | 19 | 0.1725 | **93.16%** | 10.32% | **92.17%** | 0.8886 | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` |
