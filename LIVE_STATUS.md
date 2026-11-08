# Live Pipeline Status on Jed

```text
[2026-11-08 13:32:59] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 81 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=81/81 | Clean=81/81 | GaiaNoisy=81/81
- Step 4 (5D Graphs): Clean=70/81 | GaiaNoisy=70/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40994879_[3-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (JobArrayTaskLimit)
      40994878   academic  Study2_Step5B_Train81   PD       0:00      1 (DependencyNeverSatisfied)
      40994877   academic Study2_Step4B_Graphs81    R       1:23      1 jst084
    40994879_0   academic      Study2_Step6_4Obs    R       1:23      1 jst085
    40994879_1   academic      Study2_Step6_4Obs    R       1:23      1 jst086
    40994879_2   academic      Study2_Step6_4Obs    R       1:23      1 jst089
```
