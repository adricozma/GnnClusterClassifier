# Live Pipeline Status on Jed

```text
[2026-11-06 16:36:15] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 0 / 81 ready
- Step 2 (SWIFT Runs): 0 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=0/81 | Clean=0/81 | GaiaNoisy=0/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40973245_[0-   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
    40973244   academic     Study2_Step5_Train   PD       0:00      1 (Dependency)
    40973243   academic    Study2_Step4_Graphs   PD       0:00      1 (Dependency)
40973241_[0-   academic     Study2_Step2_SWIFT   PD       0:00      1 (Dependency)
40973242_[0-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
    40973240   academic       Study2_Step1_ICs   PD       0:00      1 (Resources)
```
