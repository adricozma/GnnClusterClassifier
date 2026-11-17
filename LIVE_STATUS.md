# Live Pipeline Status on Jed

```text
[2026-11-17 23:50:00] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=81/81 | GaiaNoisy=81/81
- Step 5/6 Live GNN Training Progress:
  * 4obs_no_Dec_clean/training_metrics_gat_4obs_no_Dec_clean.log: Ep 020 | Train 0.1343 (94.71%) | Val 0.1872 (92.67%, Best=92.78%) | Test=91.48%
  * 4obs_no_Parallax_clean/training_metrics_gat_4obs_no_Parallax_clean.log: Ep 017 | Train 0.1650 (93.35%) | Val 0.2357 (90.66%, Best=91.03%) | Test=In progress
  * 4obs_no_RA_clean/training_metrics_gat_4obs_no_RA_clean.log: Ep 020 | Train 0.1632 (93.52%) | Val 0.2301 (90.57%, Best=90.59%) | Test=89.54%
  * 4obs_no_RA_noisy/training_metrics_gat_4obs_no_RA_noisy.log: Ep 011 | Train 0.3614 (83.41%) | Val 0.4919 (76.33%, Best=77.82%) | Test=76.62%
  * clean_5obs/training_metrics_gat_clean.log: Ep 020 | Train 0.1006 (96.00%) | Val 0.1495 (94.10%, Best=94.21%) | Test=93.04%
  * clean_5obs/training_metrics_gcn_clean.log: Ep 025 | Train 0.1136 (95.42%) | Val 0.1820 (92.33%, Best=92.33%) | Test=92.17%
  * noisy_5obs/training_metrics_gat_noisy.log: Ep 020 | Train 0.2683 (88.10%) | Val 0.4221 (81.26%, Best=81.76%) | Test=84.91%

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
41146835_[3-4%   academic      Study2_Step6_4Obs   PD       0:00      2 (JobArrayTaskLimit)
      41185012   academic  Study2V2_Step6_Best4D   PD       0:00      2 (Dependency)
      41185011   academic Study2V2_Step5_Train5D   PD       0:00      2 (Dependency)
      41185010   academic Study2V2_Step4_BuildGr   PD       0:00      1 (Dependency)
41185009_[0-80   academic Study2V2_Step3_Extract   PD       0:00      1 (Dependency)
      41146836   academic Study2_Step7_GCNStocha   PD       0:00      2 (Dependency)
    41146835_2   academic      Study2_Step6_4Obs    R   12:13:43      2 jst[282,294]
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4obs_no_Dec_clean` | **GAT** | 20 | 0.1343 | 94.71% | 0.1872 | 92.67% | **92.78%** | **91.48%** | 0.8828 |
| `4obs_no_Parallax_clean` | **GAT** | 17 | 0.1650 | 93.35% | 0.2357 | 90.66% | **91.03%** | **In progress** | - |
| `4obs_no_RA_clean` | **GAT** | 20 | 0.1632 | 93.52% | 0.2301 | 90.57% | **90.59%** | **89.54%** | 0.8543 |
| `4obs_no_RA_noisy` | **GAT** | 11 | 0.3614 | 83.41% | 0.4919 | 76.33% | **77.82%** | **76.62%** | 0.6238 |
| `clean_5obs` | **GAT** | 20 | 0.1006 | 96.00% | 0.1495 | 94.10% | **94.21%** | **93.04%** | 0.9035 |
| `clean_5obs` | **GCN** | 25 | 0.1136 | 95.42% | 0.1820 | 92.33% | **92.33%** | **92.17%** | 0.8886 |
| `noisy_5obs` | **GAT** | 20 | 0.2683 | 88.10% | 0.4221 | 81.26% | **81.76%** | **84.91%** | 0.7879 |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)
