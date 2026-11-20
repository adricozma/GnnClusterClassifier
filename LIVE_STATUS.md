# Live Pipeline Status on Jed

```text
[2026-11-20 17:32:05] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=81/81 | GaiaNoisy=81/81
- Step 5/6 Live GNN Training Progress:
  * 4obs_no_Dec_clean/training_metrics_gat_4obs_no_Dec_clean.log: Ep 020 | Train 0.1343 (94.71%) | Val 0.1872 (92.67%, Best=92.78%) | Test=91.48%
  * 4obs_no_Parallax_clean/training_metrics_gat_4obs_no_Parallax_clean.log: Ep 020 | Train 0.1635 (93.41%) | Val 0.2332 (90.65%, Best=91.03%) | Test=90.25%
  * 4obs_no_Parallax_noisy/training_metrics_gat_4obs_no_Parallax_noisy.log: Ep 020 | Train 0.2474 (89.07%) | Val 0.4061 (82.69%, Best=83.14%) | Test=86.72%
  * 4obs_no_RA_clean/training_metrics_gat_4obs_no_RA_clean.log: Ep 020 | Train 0.1632 (93.52%) | Val 0.2301 (90.57%, Best=90.59%) | Test=89.54%
  * 4obs_no_RA_noisy/training_metrics_gat_4obs_no_RA_noisy.log: Ep 011 | Train 0.3614 (83.41%) | Val 0.4919 (76.33%, Best=77.82%) | Test=76.62%
  * 4obs_no_pmDec_clean/training_metrics_gat_4obs_no_pmDec_clean.log: Ep 020 | Train 0.1305 (94.83%) | Val 0.1981 (91.87%, Best=91.97%) | Test=92.02%
  * 4obs_no_pmDec_noisy/training_metrics_gat_4obs_no_pmDec_noisy.log: Ep 006 | Train 0.2918 (86.53%) | Val 0.4313 (79.69%, Best=81.28%) | Test=In progress
  * 4obs_no_pmRA_clean/training_metrics_gat_4obs_no_pmRA_clean.log: Ep 020 | Train 0.1221 (95.17%) | Val 0.1851 (92.72%, Best=92.86%) | Test=92.25%
  * 4obs_no_pmRA_noisy/training_metrics_gat_4obs_no_pmRA_noisy.log: Ep 019 | Train 0.2670 (87.96%) | Val 0.4005 (81.56%, Best=81.92%) | Test=84.06%
  * clean_5obs/training_metrics_gat_clean.log: Ep 020 | Train 0.1006 (96.00%) | Val 0.1495 (94.10%, Best=94.21%) | Test=93.04%
  * clean_5obs/training_metrics_gcn_clean.log: Ep 025 | Train 0.1136 (95.42%) | Val 0.1820 (92.33%, Best=92.33%) | Test=92.17%
  * noisy_5obs/training_metrics_gat_noisy.log: Ep 020 | Train 0.2683 (88.10%) | Val 0.4221 (81.26%, Best=81.76%) | Test=84.91%

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
      41185012   academic  Study2V2_Step6_Best4D   PD       0:00      2 (Dependency)
      41185011   academic Study2V2_Step5_Train5D   PD       0:00      2 (Dependency)
      41185010   academic Study2V2_Step4_BuildGr   PD       0:00      1 (Dependency)
41185009_[0-80   academic Study2V2_Step3_Extract   PD       0:00      1 (Dependency)
      41146836   academic Study2_Step7_GCNStocha   PD       0:00      2 (Dependency)
    41146835_4   academic      Study2_Step6_4Obs    R   19:14:00      2 jst[282,294]
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4obs_no_Dec_clean` | **GAT** | 20 | 0.1343 | 94.71% | 0.1872 | 92.67% | **92.78%** | **91.48%** | 0.8828 |
| `4obs_no_Parallax_clean` | **GAT** | 20 | 0.1635 | 93.41% | 0.2332 | 90.65% | **91.03%** | **90.25%** | 0.8689 |
| `4obs_no_Parallax_noisy` | **GAT** | 20 | 0.2474 | 89.07% | 0.4061 | 82.69% | **83.14%** | **86.72%** | 0.8281 |
| `4obs_no_RA_clean` | **GAT** | 20 | 0.1632 | 93.52% | 0.2301 | 90.57% | **90.59%** | **89.54%** | 0.8543 |
| `4obs_no_RA_noisy` | **GAT** | 11 | 0.3614 | 83.41% | 0.4919 | 76.33% | **77.82%** | **76.62%** | 0.6238 |
| `4obs_no_pmDec_clean` | **GAT** | 20 | 0.1305 | 94.83% | 0.1981 | 91.87% | **91.97%** | **92.02%** | 0.8907 |
| `4obs_no_pmDec_noisy` | **GAT** | 6 | 0.2918 | 86.53% | 0.4313 | 79.69% | **81.28%** | **In progress** | - |
| `4obs_no_pmRA_clean` | **GAT** | 20 | 0.1221 | 95.17% | 0.1851 | 92.72% | **92.86%** | **92.25%** | 0.8929 |
| `4obs_no_pmRA_noisy` | **GAT** | 19 | 0.2670 | 87.96% | 0.4005 | 81.56% | **81.92%** | **84.06%** | 0.7705 |
| `clean_5obs` | **GAT** | 20 | 0.1006 | 96.00% | 0.1495 | 94.10% | **94.21%** | **93.04%** | 0.9035 |
| `clean_5obs` | **GCN** | 25 | 0.1136 | 95.42% | 0.1820 | 92.33% | **92.33%** | **92.17%** | 0.8886 |
| `noisy_5obs` | **GAT** | 20 | 0.2683 | 88.10% | 0.4221 | 81.26% | **81.76%** | **84.91%** | 0.7879 |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)
