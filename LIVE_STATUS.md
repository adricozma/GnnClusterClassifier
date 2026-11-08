# Live Pipeline Status on Jed

```text
[2026-11-08 10:05:39] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 70 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=22/81 | Clean=22/81 | GaiaNoisy=22/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40994875_[30,3   academic     Study2_Step2_SWIFT   PD       0:00      1 (Priority)
40994879_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
      40994878   academic  Study2_Step5B_Train81   PD       0:00      1 (Dependency)
      40994877   academic Study2_Step4B_Graphs81   PD       0:00      1 (Dependency)
      40994874   academic  Study2_Step5A_Train70   PD       0:00      1 (Dependency)
      40994873   academic Study2_Step4A_Graphs70   PD       0:00      1 (Dependency)
40994872_[12-2   academic       Study2_Step3_Obs   PD       0:00      1 (JobArrayTaskLimit)
40994876_[1,7-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
    40994875_1   academic     Study2_Step2_SWIFT    R       0:38      1 jst285
    40994875_7   academic     Study2_Step2_SWIFT    R       0:38      1 jst288
    40994875_8   academic     Study2_Step2_SWIFT    R       0:38      1 jst293
    40994875_9   academic     Study2_Step2_SWIFT    R       0:38      1 jst306
   40994875_25   academic     Study2_Step2_SWIFT    R       0:38      1 jst315
    40994872_0   academic       Study2_Step3_Obs    R       0:38      1 jst089
   40994872_10   academic       Study2_Step3_Obs    R       0:38      1 jst299
```
