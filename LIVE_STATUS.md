# Live Pipeline Status on Jed

```text
[2026-11-08 11:21:26] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 74 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=70/81 | Clean=70/81 | GaiaNoisy=70/81
- Step 4 (5D Graphs): Clean=57/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40994875_[57,6   academic     Study2_Step2_SWIFT   PD       0:00      1 (JobArrayTaskLimit)
40994879_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
      40994878   academic  Study2_Step5B_Train81   PD       0:00      1 (Dependency)
      40994877   academic Study2_Step4B_Graphs81   PD       0:00      1 (Dependency)
      40994874   academic  Study2_Step5A_Train70   PD       0:00      1 (Dependency)
40994876_[1,7-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
   40994875_47   academic     Study2_Step2_SWIFT    R       9:55      1 jst288
   40994875_43   academic     Study2_Step2_SWIFT    R      25:55      1 jst315
      40994873   academic Study2_Step4A_Graphs70    R       5:55      1 jst298
   40994875_32   academic     Study2_Step2_SWIFT    R      30:56      1 jst285
   40994875_30   academic     Study2_Step2_SWIFT    R      52:57      1 jst277
    40994875_8   academic     Study2_Step2_SWIFT    R    1:14:59      1 jst293
    40994875_9   academic     Study2_Step2_SWIFT    R    1:14:59      1 jst306
```
