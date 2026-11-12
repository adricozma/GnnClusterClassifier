# Live Pipeline Status on Jed

```text
[2026-11-12 19:37:43] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=81/81 | GaiaNoisy=81/81
- Step 5/6 Live GNN Training Progress:
  * clean_5obs/training_metrics_gat_clean.log: Ep 006 | Train 0.2081 (91.25%) | Val 0.2562 (89.37%, Best=89.37%) | CoreFNR: 10.32% (min=10.32%) | Test=In progress
  * clean_5obs/training_metrics_gcn_clean.log: Ep 025 | Train 0.1136 (95.42%) | Val 0.1820 (92.33%, Best=92.33%) | Test=92.17%
  * legacy/clean_5obs_static_5rot/training_metrics_gcn_clean.log: Ep 025 | Train 0.1136 (95.42%) | Val 0.1820 (92.33%, Best=92.33%) | Test=92.17%
- Saved Model Checkpoints (.pth): 8 mirrored in GNN-jed/trained_models/study2/weights/

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
      41146836   academic Study2_Step7_GCNStocha   PD       0:00      2 (Dependency)
41146835_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      2 (Dependency)
      41146834   academic   Study2_Step5_Train81    R    4:53:23      2 jst[317,320]
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Core FNR | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_5obs` | **GAT** | 6 | 0.2081 | 91.25% | 0.2562 | 89.37% | **89.37%** | 10.32% (val min) | **In progress** | - |
| `clean_5obs` | **GCN** | 25 | 0.1136 | 95.42% | 0.1820 | 92.33% | **92.33%** | - | **92.17%** | 0.8886 |
| `clean_5obs_static_5rot` | **GCN** | 25 | 0.1136 | 95.42% | 0.1820 | 92.33% | **92.33%** | - | **92.17%** | 0.8886 |

## Saved Model Weights (`.pth` Checkpoints)

| Checkpoint | Size | Experiment Path | Central Archive Mirror |
| :--- | :---: | :--- | :--- |
| `best_gat_clean.pth` | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean.pth` |
| `best_gat_clean_ep18_backup.pth` | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean_ep18_backup.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean_ep18_backup.pth` |
| `best_gcn_clean.pth` | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gcn_clean.pth` |
| `best_gat_clean_ep18_backup.pth` | 361.2 KB | `GNN-jed/trained_models/study2/legacy/clean_5obs_static_5rot/best_gat_clean_ep18_backup.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean_ep18_backup.pth` |
| `best_gcn_clean.pth` | 120.0 KB | `GNN-jed/trained_models/study2/legacy/clean_5obs_static_5rot/best_gcn_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gcn_clean.pth` |
| `best_gat_clean.pth` | 361.2 KB | `GNN-jed/trained_models/study2/legacy/weights/best_gat_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean.pth` |
| `best_gat_clean_ep18_backup.pth` | 361.2 KB | `GNN-jed/trained_models/study2/legacy/weights/best_gat_clean_ep18_backup.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean_ep18_backup.pth` |
| `best_gcn_clean.pth` | 120.0 KB | `GNN-jed/trained_models/study2/legacy/weights/best_gcn_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gcn_clean.pth` |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)
