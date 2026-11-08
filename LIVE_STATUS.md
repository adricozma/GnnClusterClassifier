# Live Pipeline Status on Jed

```text
[2026-11-08 10:04:06] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 70 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=22/81 | Clean=22/81 | GaiaNoisy=22/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40994879_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
      40994878   academic  Study2_Step5B_Train81   PD       0:00      1 (Dependency)
      40994877   academic Study2_Step4B_Graphs81   PD       0:00      1 (Dependency)
40994875_[1,7-   academic     Study2_Step2_SWIFT   PD       0:00      1 (Priority)
      40994874   academic  Study2_Step5A_Train70   PD       0:00      1 (Dependency)
      40994873   academic Study2_Step4A_Graphs70   PD       0:00      1 (Dependency)
40994876_[1,7-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
40994872_[0,2-   academic       Study2_Step3_Obs   PD       0:00      1 (Priority)
```
