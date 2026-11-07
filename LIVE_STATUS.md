# Live Pipeline Status on Jed

```text
[2026-11-07 14:16:30] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 77 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=36/81 | Clean=22/81 | GaiaNoisy=22/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40984797_[0-   academic      Study2_Step6_4Obs   PD       0:00      1 (DependencyNeverSatisfied)
    40984796   academic  Study2_Step5B_Train81   PD       0:00      1 (Dependency)
    40984795   academic Study2_Step4B_Graphs81   PD       0:00      1 (DependencyNeverSatisfied)
    40984784   academic  Study2_Step5A_Train75   PD       0:00      1 (Dependency)
    40984783   academic Study2_Step4A_Graphs75   PD       0:00      1 (DependencyNeverSatisfied)
40984786_[1,   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
  40984785_8   academic     Study2_Step2_SWIFT    R    1:05:33      1 jst316
 40984785_43   academic     Study2_Step2_SWIFT    R    1:05:33      1 jst319
```
