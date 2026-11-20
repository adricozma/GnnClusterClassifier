# Live Pipeline Status on Jed

```text
[2026-11-20 21:11:08] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=81/81 | GaiaNoisy=81/81
- Step 5/6 Live GNN Training Progress:
  * 4obs_no_Dec_clean/training_metrics_gat_4obs_no_Dec_clean.log: Ep 020 | Train 0.1343 (94.71%) | Val 0.1872 (92.67%, Best=92.78%) | CoreFNR: 6.17% (min=5.48%) | Test=91.48%
  * 4obs_no_Parallax_clean/training_metrics_gat_4obs_no_Parallax_clean.log: Ep 020 | Train 0.1635 (93.41%) | Val 0.2332 (90.65%, Best=91.03%) | CoreFNR: 6.15% (min=5.63%) | Test=90.25%
  * 4obs_no_Parallax_noisy/training_metrics_gat_4obs_no_Parallax_noisy.log: Ep 020 | Train 0.2474 (89.07%) | Val 0.4061 (82.69%, Best=83.14%) | CoreFNR: 11.39% (min=7.78%) | Test=86.72%
  * 4obs_no_RA_clean/training_metrics_gat_4obs_no_RA_clean.log: Ep 020 | Train 0.1632 (93.52%) | Val 0.2301 (90.57%, Best=90.59%) | CoreFNR: 11.50% (min=8.41%) | Test=89.54%
  * 4obs_no_RA_noisy/training_metrics_gat_4obs_no_RA_noisy.log: Ep 011 | Train 0.3614 (83.41%) | Val 0.4919 (76.33%, Best=77.82%) | CoreFNR: 36.76% (min=10.87%) | Test=76.62%
  * 4obs_no_pmDec_clean/training_metrics_gat_4obs_no_pmDec_clean.log: Ep 020 | Train 0.1305 (94.83%) | Val 0.1981 (91.87%, Best=91.97%) | CoreFNR: 9.29% (min=7.19%) | Test=92.02%
  * 4obs_no_pmDec_noisy/training_metrics_gat_4obs_no_pmDec_noisy.log: Ep 011 | Train 0.2727 (87.63%) | Val 0.3809 (81.45%, Best=81.52%) | CoreFNR: 27.37% (min=12.74%) | Test=In progress
  * 4obs_no_pmRA_clean/training_metrics_gat_4obs_no_pmRA_clean.log: Ep 020 | Train 0.1221 (95.17%) | Val 0.1851 (92.72%, Best=92.86%) | CoreFNR: 8.40% (min=6.92%) | Test=92.25%
  * 4obs_no_pmRA_noisy/training_metrics_gat_4obs_no_pmRA_noisy.log: Ep 019 | Train 0.2670 (87.96%) | Val 0.4005 (81.56%, Best=81.92%) | CoreFNR: 20.42% (min=13.75%) | Test=84.06%
  * clean_5obs/training_metrics_gat_clean.log: Ep 020 | Train 0.1006 (96.00%) | Val 0.1495 (94.10%, Best=94.21%) | CoreFNR: 6.64% (min=6.60%) | Test=93.04%
  * clean_5obs/training_metrics_gcn_clean.log: Ep 025 | Train 0.1136 (95.42%) | Val 0.1820 (92.33%, Best=92.33%) | Test=92.17%
  * noisy_5obs/training_metrics_gat_noisy.log: Ep 020 | Train 0.2683 (88.10%) | Val 0.4221 (81.26%, Best=81.76%) | CoreFNR: 23.60% (min=14.33%) | Test=84.91%
  * [IN FLIGHT] 4obs_no_pmDec_noisy (GAT): Ep 011 | Batch 0103/0103 (100.0%) | Batch Loss 0.2632 | LR: 2.17e-04 | RAM: 113.0G
- Saved Model Checkpoints (.pth): 12 mirrored in GNN-jed/trained_models/study2/weights/

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
      41185012   academic  Study2V2_Step6_Best4D   PD       0:00      2 (Dependency)
      41185011   academic Study2V2_Step5_Train5D   PD       0:00      2 (Dependency)
      41185010   academic Study2V2_Step4_BuildGr   PD       0:00      1 (Dependency)
41185009_[0-80   academic Study2V2_Step3_Extract   PD       0:00      1 (Dependency)
      41146836   academic Study2_Step7_GCNStocha   PD       0:00      2 (Dependency)
    41146835_4   academic      Study2_Step6_4Obs    R   22:51:41      2 jst[282,294]
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Core FNR | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4obs_no_Dec_clean` | **GAT** | 20 | 0.1343 | 94.71% | 0.1872 | 92.67% | **92.78%** | 7.90% (val: 5.48%) | **91.48%** | 0.8828 |
| `4obs_no_Parallax_clean` | **GAT** | 20 | 0.1635 | 93.41% | 0.2332 | 90.65% | **91.03%** | 7.27% (val: 5.63%) | **90.25%** | 0.8689 |
| `4obs_no_Parallax_noisy` | **GAT** | 20 | 0.2474 | 89.07% | 0.4061 | 82.69% | **83.14%** | 8.14% (val: 7.78%) | **86.72%** | 0.8281 |
| `4obs_no_RA_clean` | **GAT** | 20 | 0.1632 | 93.52% | 0.2301 | 90.57% | **90.59%** | 12.00% (val: 8.41%) | **89.54%** | 0.8543 |
| `4obs_no_RA_noisy` | **GAT** | 11 | 0.3614 | 83.41% | 0.4919 | 76.33% | **77.82%** | 44.35% (val: 10.87%) | **76.62%** | 0.6238 |
| `4obs_no_pmDec_clean` | **GAT** | 20 | 0.1305 | 94.83% | 0.1981 | 91.87% | **91.97%** | 6.67% (val: 7.19%) | **92.02%** | 0.8907 |
| `4obs_no_pmDec_noisy` | **GAT** | 11 | 0.2727 | 87.63% | 0.3809 | 81.45% | **81.52%** | 12.74% (val min) | **In progress** | - |
| `4obs_no_pmRA_clean` | **GAT** | 20 | 0.1221 | 95.17% | 0.1851 | 92.72% | **92.86%** | 7.20% (val: 6.92%) | **92.25%** | 0.8929 |
| `4obs_no_pmRA_noisy` | **GAT** | 19 | 0.2670 | 87.96% | 0.4005 | 81.56% | **81.92%** | 23.21% (val: 13.75%) | **84.06%** | 0.7705 |
| `clean_5obs` | **GAT** | 20 | 0.1006 | 96.00% | 0.1495 | 94.10% | **94.21%** | 6.49% (val: 6.60%) | **93.04%** | 0.9035 |
| `clean_5obs` | **GCN** | 25 | 0.1136 | 95.42% | 0.1820 | 92.33% | **92.33%** | - | **92.17%** | 0.8886 |
| `noisy_5obs` | **GAT** | 20 | 0.2683 | 88.10% | 0.4221 | 81.26% | **81.76%** | 19.52% (val: 14.33%) | **84.91%** | 0.7879 |
| `4obs_no_pmDec_noisy` | **GAT (In Flight)** | Ep 11 (100%) | 0.2632 | - | - | - | - | - | **In progress** | - |

## Saved Model Weights (`.pth` Checkpoints)

| Checkpoint | Size | Experiment Path | Central Archive Mirror |
| :--- | :---: | :--- | :--- |
| `best_gat_4obs_no_Dec_clean.pth` | 358.0 KB | `GNN-jed/trained_models/study2/4obs_no_Dec_clean/best_gat_4obs_no_Dec_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_Dec_clean.pth` |
| `best_gat_4obs_no_Parallax_clean.pth` | 358.1 KB | `GNN-jed/trained_models/study2/4obs_no_Parallax_clean/best_gat_4obs_no_Parallax_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_Parallax_clean.pth` |
| `best_gat_4obs_no_Parallax_noisy.pth` | 358.1 KB | `GNN-jed/trained_models/study2/4obs_no_Parallax_noisy/best_gat_4obs_no_Parallax_noisy.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_Parallax_noisy.pth` |
| `best_gat_4obs_no_RA_clean.pth` | 357.9 KB | `GNN-jed/trained_models/study2/4obs_no_RA_clean/best_gat_4obs_no_RA_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_RA_clean.pth` |
| `best_gat_4obs_no_RA_noisy.pth` | 357.9 KB | `GNN-jed/trained_models/study2/4obs_no_RA_noisy/best_gat_4obs_no_RA_noisy.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_RA_noisy.pth` |
| `best_gat_4obs_no_pmDec_clean.pth` | 358.0 KB | `GNN-jed/trained_models/study2/4obs_no_pmDec_clean/best_gat_4obs_no_pmDec_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_pmDec_clean.pth` |
| `best_gat_4obs_no_pmDec_noisy.pth` | 358.0 KB | `GNN-jed/trained_models/study2/4obs_no_pmDec_noisy/best_gat_4obs_no_pmDec_noisy.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_pmDec_noisy.pth` |
| `best_gat_4obs_no_pmRA_clean.pth` | 358.0 KB | `GNN-jed/trained_models/study2/4obs_no_pmRA_clean/best_gat_4obs_no_pmRA_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_pmRA_clean.pth` |
| `best_gat_4obs_no_pmRA_noisy.pth` | 358.0 KB | `GNN-jed/trained_models/study2/4obs_no_pmRA_noisy/best_gat_4obs_no_pmRA_noisy.pth` | `GNN-jed/trained_models/study2/weights/best_gat_4obs_no_pmRA_noisy.pth` |
| `best_gat_clean.pth` | 361.2 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gat_clean.pth` |
| `best_gcn_clean.pth` | 120.0 KB | `GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth` | `GNN-jed/trained_models/study2/weights/best_gcn_clean.pth` |
| `best_gat_noisy.pth` | 361.2 KB | `GNN-jed/trained_models/study2/noisy_5obs/best_gat_noisy.pth` | `GNN-jed/trained_models/study2/weights/best_gat_noisy.pth` |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)

## RMT Singular Value Candle Spectra (Marchenko-Pastur Bulk vs Detached Signal)

### `4obs_no_Dec_clean`
![RMT Candle Spectrum 4obs_no_Dec_clean](GNN-jed/trained_models/study2/4obs_no_Dec_clean/spectral_analysis_GNN_4obs_no_Dec_clean.png)

### `4obs_no_Parallax_clean`
![RMT Candle Spectrum 4obs_no_Parallax_clean](GNN-jed/trained_models/study2/4obs_no_Parallax_clean/spectral_analysis_GNN_4obs_no_Parallax_clean.png)

### `4obs_no_Parallax_noisy`
![RMT Candle Spectrum 4obs_no_Parallax_noisy](GNN-jed/trained_models/study2/4obs_no_Parallax_noisy/spectral_analysis_GNN_4obs_no_Parallax_noisy.png)

### `4obs_no_RA_clean`
![RMT Candle Spectrum 4obs_no_RA_clean](GNN-jed/trained_models/study2/4obs_no_RA_clean/spectral_analysis_GNN_4obs_no_RA_clean.png)

### `4obs_no_RA_noisy`
![RMT Candle Spectrum 4obs_no_RA_noisy](GNN-jed/trained_models/study2/4obs_no_RA_noisy/spectral_analysis_GNN_4obs_no_RA_noisy.png)

### `4obs_no_pmDec_clean`
![RMT Candle Spectrum 4obs_no_pmDec_clean](GNN-jed/trained_models/study2/4obs_no_pmDec_clean/spectral_analysis_GNN_4obs_no_pmDec_clean.png)

### `4obs_no_pmDec_noisy`
![RMT Candle Spectrum 4obs_no_pmDec_noisy](GNN-jed/trained_models/study2/4obs_no_pmDec_noisy/spectral_analysis_GNN_4obs_no_pmDec_noisy.png)

### `4obs_no_pmRA_clean`
![RMT Candle Spectrum 4obs_no_pmRA_clean](GNN-jed/trained_models/study2/4obs_no_pmRA_clean/spectral_analysis_GNN_4obs_no_pmRA_clean.png)

### `4obs_no_pmRA_noisy`
![RMT Candle Spectrum 4obs_no_pmRA_noisy](GNN-jed/trained_models/study2/4obs_no_pmRA_noisy/spectral_analysis_GNN_4obs_no_pmRA_noisy.png)

### `clean_5obs`
![RMT Candle Spectrum clean_5obs](GNN-jed/trained_models/study2/clean_5obs/spectral_analysis_GNN_clean_5obs.png)

### `noisy_5obs`
![RMT Candle Spectrum noisy_5obs](GNN-jed/trained_models/study2/noisy_5obs/spectral_analysis_GNN_noisy_5obs.png)

