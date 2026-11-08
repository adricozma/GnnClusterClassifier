# Live Pipeline Status on Jed

```text
[2026-11-08 10:30:32] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 70 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=70/81 | Clean=70/81 | GaiaNoisy=70/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40994875_[32,4   academic     Study2_Step2_SWIFT   PD       0:00      1 (JobArrayTaskLimit)
      40994873   academic Study2_Step4A_Graphs70   PD       0:00      1 (Priority)
40994879_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
      40994878   academic  Study2_Step5B_Train81   PD       0:00      1 (Dependency)
      40994877   academic Study2_Step4B_Graphs81   PD       0:00      1 (Dependency)
      40994874   academic  Study2_Step5A_Train70   PD       0:00      1 (Dependency)
40994876_[1,7-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
   40994875_30   academic     Study2_Step2_SWIFT    R       4:04      1 jst277
    40994875_1   academic     Study2_Step2_SWIFT    R      26:06      1 jst285
    40994875_7   academic     Study2_Step2_SWIFT    R      26:06      1 jst288
    40994875_8   academic     Study2_Step2_SWIFT    R      26:06      1 jst293
    40994875_9   academic     Study2_Step2_SWIFT    R      26:06      1 jst306
   40994875_25   academic     Study2_Step2_SWIFT    R      26:06      1 jst315
```
