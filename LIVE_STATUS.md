# Live Pipeline Status on Jed

```text
[2026-11-08 20:15:43] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=81/81 | GaiaNoisy=81/81
- Step 5/6 Live GNN Training Progress:
  * clean_5obs/training_metrics_gcn_clean.log: Ep 004 | Train 0.1602 (93.36%) | Val 0.2139 (91.04%, Best=91.04%) | Test=In progress

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40998767_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      2 (Dependency)
      40998766   academic   Study2_Step5_Train81    R    3:45:34      2 jst[072,076]
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_5obs` | **GCN** | 4 | 0.1602 | 93.36% | 0.2139 | 91.04% | **91.04%** | **In progress** | - |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)
