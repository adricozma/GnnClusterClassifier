# Study 2 Trained GNN Weights Registry

All best model checkpoints (`.pth`) are saved both in their respective experiment directories and mirrored in `GNN-jed/trained_models/study2/weights/`.

| Checkpoint File | Architecture | Features | Best Epoch | Val Loss | Val Acc | Core FNR | Test Acc | Test F1 | Size | Experiment Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `best_gat_clean.pth` | **GAT** | 5D | 35 | 0.2071 | **91.89%** | 9.22% | **91.43%** | 0.8807 | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` |
| `best_gat_noisy.pth` | **GAT** | 5D | 6 | 0.4108 | **81.00%** | 22.08% | **Pending** | - | 361.2 KB | `GNN-jed/trained_models/study2/noisy_5obs/best_gat_noisy.pth` |
| `best_gcn_clean.pth` | **GCN** | 5D | 24 | 0.1817 | **92.32%** | - | **92.17%** | 0.8886 | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` |
