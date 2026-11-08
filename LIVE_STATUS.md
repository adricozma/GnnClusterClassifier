# Live Pipeline Status on Jed

```text
[2026-11-08 12:06:59] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 78 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=70/81 | Clean=70/81 | GaiaNoisy=70/81
- Step 4 (5D Graphs): Clean=70/81 | GaiaNoisy=70/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40994879_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (Dependency)
      40994878   academic  Study2_Step5B_Train81   PD       0:00      1 (DependencyNeverSatisfied)
      40994877   academic Study2_Step4B_Graphs81   PD       0:00      1 (Dependency)
40994876_[1,7-   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
   40994875_57   academic     Study2_Step2_SWIFT    R      41:23      1 jst306
   40994875_62   academic     Study2_Step2_SWIFT    R      32:23      1 jst285
   40994875_47   academic     Study2_Step2_SWIFT    R      54:24      1 jst288
   40994875_43   academic     Study2_Step2_SWIFT    R    1:10:24      1 jst315
```
