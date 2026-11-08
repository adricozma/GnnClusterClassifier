# Live Pipeline Status on Jed

```text
[2026-11-08 11:30:56] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 75 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=70/81 | Clean=70/81 | GaiaNoisy=70/81
- Step 4 (5D Graphs): Clean=70/81 | GaiaNoisy=44/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
   40994875_62   academic     Study2_Step2_SWIFT   PD       0:00      1 (JobArrayTaskLimit)
40994879_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
      40994878   academic  Study2_Step5B_Train81   PD       0:00      1 (Dependency)
      40994877   academic Study2_Step4B_Graphs81   PD       0:00      1 (Dependency)
      40994874   academic  Study2_Step5A_Train70   PD       0:00      1 (Dependency)
40994876_[1,7-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
   40994875_57   academic     Study2_Step2_SWIFT    R       6:58      1 jst306
   40994875_47   academic     Study2_Step2_SWIFT    R      19:59      1 jst288
   40994875_43   academic     Study2_Step2_SWIFT    R      35:59      1 jst315
      40994873   academic Study2_Step4A_Graphs70    R      15:59      1 jst298
   40994875_32   academic     Study2_Step2_SWIFT    R      41:00      1 jst285
   40994875_30   academic     Study2_Step2_SWIFT    R    1:03:01      1 jst277
    40994875_8   academic     Study2_Step2_SWIFT    R    1:25:03      1 jst293
```
