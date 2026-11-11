# Live Pipeline Status on Jed

```text
[2026-11-11 13:05:35] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 0 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=0/81 | Clean=0/81 | GaiaNoisy=0/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81
- Step 5/6 Live GNN Training Progress:
  * clean_5obs/training_metrics_gat_clean.log: Ep 018 | Train 0.1037 (95.87%) | Val 0.1470 (94.20%, Best=94.20%) | Test=In progress
  * clean_5obs/training_metrics_gcn_clean.log: Ep 025 | Train 0.1136 (95.42%) | Val 0.1820 (92.33%, Best=92.33%) | Test=92.17%
- Saved Model Checkpoints (.pth): 2 mirrored in GNN-jed/trained_models/study2/weights/

SLURM Queue:
Unable to query squeue: Command '['squeue', '-u', 'cozma', '-o', '%.14i %.10P %.22j %.4t %.10M %.6D %R']' timed out after 10 seconds
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Core FNR | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_5obs` | **GAT** | 18 | 0.1037 | 95.87% | 0.1470 | 94.20% | **94.20%** | - | **In progress** | - |
| `clean_5obs` | **GCN** | 25 | 0.1136 | 95.42% | 0.1820 | 92.33% | **92.33%** | - | **92.17%** | 0.8886 |

## Saved Model Weights (`.pth` Checkpoints)

| Checkpoint | Size | Experiment Path | Central Archive Mirror |
| :--- | :---: | :--- | :--- |
| `best_gat_clean.pth` | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean.pth` |
| `best_gcn_clean.pth` | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gcn_clean.pth` |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)
