# Live Pipeline Status on Jed

```text
[2026-11-14 18:33:40] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=81/81 | GaiaNoisy=81/81
- Step 5/6 Live GNN Training Progress:
  * 4obs_no_RA_clean/training_metrics_gat_4obs_no_RA_clean.log: Ep 002 | Train 0.1752 (92.98%) | Val 0.2404 (90.04%, Best=90.04%) | CoreFNR: 15.56% (min=8.41%) | Test=In progress
  * clean_5obs/training_metrics_gat_clean.log: Ep 020 | Train 0.1006 (96.00%) | Val 0.1495 (94.10%, Best=94.21%) | CoreFNR: 6.64% (min=6.60%) | Test=93.04%
  * clean_5obs/training_metrics_gcn_clean.log: Ep 025 | Train 0.1136 (95.42%) | Val 0.1820 (92.33%, Best=92.33%) | Test=92.17%
  * noisy_5obs/training_metrics_gat_noisy.log: Ep 020 | Train 0.2683 (88.10%) | Val 0.4221 (81.26%, Best=81.76%) | CoreFNR: 23.60% (min=14.33%) | Test=84.91%
  * [IN FLIGHT] 4obs_no_RA_clean (GAT): Ep 002 | Batch 0103/0103 (100.0%) | Batch Loss 0.0767 | LR: 1.83e-04 | RAM: 1.4G
- Saved Model Checkpoints (.pth): 4 mirrored in GNN-jed/trained_models/study2/weights/

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
41146835_[1-4%   academic      Study2_Step6_4Obs   PD       0:00      2 (JobArrayTaskLimit)
      41185012   academic  Study2V2_Step6_Best4D   PD       0:00      2 (Dependency)
      41185011   academic Study2V2_Step5_Train5D   PD       0:00      2 (Dependency)
      41146836   academic Study2_Step7_GCNStocha   PD       0:00      2 (Dependency)
      41185010   academic Study2V2_Step4_BuildGr   PD       0:00      1 (Dependency)
41185009_[0-80   academic Study2V2_Step3_Extract   PD       0:00      1 (Dependency)
    41146835_0   academic      Study2_Step6_4Obs    R    1:36:27      2 jst[292,312]
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Core FNR | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4obs_no_RA_clean` | **GAT** | 2 | 0.1752 | 92.98% | 0.2404 | 90.04% | **90.04%** | 8.41% (val min) | **In progress** | - |
| `clean_5obs` | **GAT** | 20 | 0.1006 | 96.00% | 0.1495 | 94.10% | **94.21%** | 6.49% (val: 6.60%) | **93.04%** | 0.9035 |
| `clean_5obs` | **GCN** | 25 | 0.1136 | 95.42% | 0.1820 | 92.33% | **92.33%** | - | **92.17%** | 0.8886 |
| `noisy_5obs` | **GAT** | 20 | 0.2683 | 88.10% | 0.4221 | 81.26% | **81.76%** | 19.52% (val: 14.33%) | **84.91%** | 0.7879 |
| `4obs_no_RA_clean` | **GAT (In Flight)** | Ep 02 (100%) | 0.0767 | - | - | - | - | - | **In progress** | - |

## Saved Model Weights (`.pth` Checkpoints)

| Checkpoint | Size | Experiment Path | Central Archive Mirror |
| :--- | :---: | :--- | :--- |
| `best_gat_4obs_no_RA_clean.pth` | 357.9 KB | `GNN-jed/trained_models/study2/4obs_no_RA_clean/best_gat_4obs_no_RA_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_RA_clean.pth` |
| `best_gat_clean.pth` | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean.pth` |
| `best_gcn_clean.pth` | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gcn_clean.pth` |
| `best_gat_noisy.pth` | 361.2 KB | `GNN-jed/trained_models/study2/noisy_5obs/best_gat_noisy.pth` | `GNN-jed/trained_models/study2/weights/best_gat_noisy.pth` |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)

## RMT Singular Value Candle Spectra (Marchenko-Pastur Bulk vs Detached Signal)

### `4obs_no_RA_clean`
![RMT Candle Spectrum 4obs_no_RA_clean](GNN-jed/trained_models/study2/4obs_no_RA_clean/spectral_analysis_GNN_4obs_no_RA_clean.png)

### `clean_5obs`
![RMT Candle Spectrum clean_5obs](GNN-jed/trained_models/study2/clean_5obs/spectral_analysis_GNN_clean_5obs.png)

### `noisy_5obs`
![RMT Candle Spectrum noisy_5obs](GNN-jed/trained_models/study2/noisy_5obs/spectral_analysis_GNN_noisy_5obs.png)

