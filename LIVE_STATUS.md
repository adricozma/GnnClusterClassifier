# Live Pipeline Status on Jed

```text
[2026-11-07 13:10:59] Study 2 Pipeline Status on Jed
- Step 1 (Plummer ICs): 81 / 81 ready
- Step 2 (SWIFT Runs): 75 / 81 completed (101 snapshots)
- Step 3 (Observables): Labels=0/81 | Clean=0/81 | GaiaNoisy=0/81
- Step 4 (5D Graphs): Clean=0/81 | GaiaNoisy=0/81

SLURM Queue:
JOBID  PARTITION                   NAME   ST       TIME  NODES NODELIST(REASON)
40984797_[0-4%   academic      Study2_Step6_4Obs   PD       0:00      1 (None)
      40984796   academic  Study2_Step5B_Train81   PD       0:00      1 (Dependency)
      40984795   academic Study2_Step4B_Graphs81   PD       0:00      1 (Dependency)
      40984784   academic  Study2_Step5A_Train75   PD       0:00      1 (Dependency)
      40984783   academic Study2_Step4A_Graphs75   PD       0:00      1 (Dependency)
40984782_[4-7,   academic       Study2_Step3_Obs   PD       0:00      1 (Resources)
40984786_[1,8,   academic       Study2_Step3_Obs   PD       0:00      1 (Dependency)
    40984785_1   academic     Study2_Step2_SWIFT    R       0:01      1 jst314
    40984785_8   academic     Study2_Step2_SWIFT    R       0:01      1 jst316
   40984785_25   academic     Study2_Step2_SWIFT    R       0:01      1 jst317
   40984785_43   academic     Study2_Step2_SWIFT    R       0:01      1 jst319
   40984785_47   academic     Study2_Step2_SWIFT    R       0:01      1 jst321
   40984785_62   academic     Study2_Step2_SWIFT    R       0:01      1 jst322
    40984782_0   academic       Study2_Step3_Obs    R       0:01      1 jst277
    40984782_2   academic       Study2_Step3_Obs    R       0:01      1 jst304
    40984782_3   academic       Study2_Step3_Obs    R       0:01      1 jst319
```
