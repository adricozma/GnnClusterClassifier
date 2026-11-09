# Live Pipeline Status on Jed

```text
[2026-11-09 04:51:18] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=81/81 | GaiaNoisy=81/81
- Step 5/6 Live GNN Training Progress:
  * clean_5obs/training_metrics_gcn_clean.log: Ep 014 | Train 0.1221 (95.07%) | Val 0.1964 (91.58%, Best=91.79%) | Test=In progress

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40998767_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      2 (Dependency)
      40998766   academic   Study2_Step5_Train81    R   12:21:30      2 jst[072,076]
```

## Live Neural Network Training Metrics

| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Test Acc | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_5obs` | **GCN** | 14 | 0.1221 | 95.07% | 0.1964 | 91.58% | **91.79%** | **In progress** | - |

## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)

![Live GNN Training Curves](GNN-jed/trained_models/study2/latest_live_training.png)
