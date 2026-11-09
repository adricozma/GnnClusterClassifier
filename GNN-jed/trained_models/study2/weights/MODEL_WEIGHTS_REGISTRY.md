# Study 2 Trained GNN Weights Registry

All best model checkpoints (`.pth`) are saved both in their respective experiment directories and mirrored in `GNN-jed/trained_models/study2/weights/`.

| Checkpoint File | Architecture | Features | Best Epoch | Val Loss | Val Acc | Test Acc | Test F1 | Size | Experiment Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `best_gat_clean.pth` | **GAT** | 5D | 2 | 0.2250 | **91.08%** | **Pending** | - | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` |
| `best_gcn_clean.pth` | **GCN** | 5D | 24 | 0.1817 | **92.32%** | **92.17%** | 0.8886 | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` |
