# Study 2 Trained GNN Weights Registry

All best model checkpoints (`.pth`) are saved both in their respective experiment directories and mirrored in `GNN-jed/trained_models/study2/weights/`.

| Checkpoint File | Architecture | Features | Best Epoch | Val Loss | Val Acc | Core FNR | Test Acc | Test F1 | Size | Experiment Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `best_gat_4obs_no_RA_clean.pth` | **GAT** | 4D | 2 | 0.2404 | **90.04%** | 15.56% | **Pending** | - | 357.9 KB | `GNN-jed/trained_models/study2/4obs_no_RA_clean/best_gat_4obs_no_RA_clean.pth` |
| `best_gat_clean.pth` | **GAT** | 5D | 18 | 0.1470 | **94.20%** | 6.49% | **93.04%** | 0.9035 | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` |
| `best_gat_noisy.pth` | **GAT** | 5D | 12 | 0.4014 | **81.76%** | 19.52% | **84.91%** | 0.7879 | 361.2 KB | `GNN-jed/trained_models/study2/noisy_5obs/best_gat_noisy.pth` |
| `best_gcn_clean.pth` | **GCN** | 5D | 24 | 0.1817 | **92.32%** | 10.32% | **92.17%** | 0.8886 | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` |
