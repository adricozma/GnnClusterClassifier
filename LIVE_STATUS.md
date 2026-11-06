# Live Pipeline Status on Jed

```text
[2026-11-06 16:15:11] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 0 / 81 ready
- Step 2 (SWIFT Runs): 0 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=0/81 | Clean=0/81 | GaiaNoisy=0/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
    40972016   academic       Study2_Step1_ICs   PD       0:00      1 (QOSGrpCpuLimit)
40972021_[0-   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
    40972020   academic     Study2_Step5_Train   PD       0:00      1 (Dependency)
    40972019   academic    Study2_Step4_Graphs   PD       0:00      1 (Dependency)
40972018_[0-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
40972017_[0-   academic     Study2_Step2_SWIFT   PD       0:00      1 (Dependency)
```
