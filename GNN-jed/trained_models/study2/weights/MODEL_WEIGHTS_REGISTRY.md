# Study 2 Trained GNN Weights Registry

All best model checkpoints (`.pth`) are saved both in their respective experiment directories and mirrored in `GNN-jed/trained_models/study2/weights/`.

| Checkpoint File | Architecture | Features | Best Epoch | Val Loss | Val Acc | Core FNR | Test Acc | Test F1 | Size | Experiment Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `best_gat_4obs_no_Dec_clean.pth` | **GAT** | 4D | 13 | 0.1803 | **92.78%** | 7.90% | **91.48%** | 0.8828 | 358.0 KB | `GNN-jed/trained_models/study2/4obs_no_Dec_clean/best_gat_4obs_no_Dec_clean.pth` |
| `best_gat_4obs_no_Parallax_clean.pth` | **GAT** | 4D | 13 | 0.2208 | **90.96%** | 7.58% | **Pending** | - | 358.1 KB | `GNN-jed/trained_models/study2/4obs_no_Parallax_clean/best_gat_4obs_no_Parallax_clean.pth` |
| `best_gat_4obs_no_RA_clean.pth` | **GAT** | 4D | 16 | 0.2292 | **90.59%** | 12.00% | **89.54%** | 0.8543 | 357.9 KB | `GNN-jed/trained_models/study2/4obs_no_RA_clean/best_gat_4obs_no_RA_clean.pth` |
| `best_gat_4obs_no_RA_noisy.pth` | **GAT** | 4D | 3 | 0.4650 | **77.82%** | 44.35% | **76.62%** | 0.6238 | 357.9 KB | `GNN-jed/trained_models/study2/4obs_no_RA_noisy/best_gat_4obs_no_RA_noisy.pth` |
| `best_gat_clean.pth` | **GAT** | 5D | 18 | 0.1470 | **94.20%** | 6.49% | **93.04%** | 0.9035 | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` |
| `best_gat_noisy.pth` | **GAT** | 5D | 12 | 0.4014 | **81.76%** | 19.52% | **84.91%** | 0.7879 | 361.2 KB | `GNN-jed/trained_models/study2/noisy_5obs/best_gat_noisy.pth` |
| `best_gcn_clean.pth` | **GCN** | 5D | 24 | 0.1817 | **92.32%** | 10.32% | **92.17%** | 0.8886 | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` |
