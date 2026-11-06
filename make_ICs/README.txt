Author: Adrian Cozma (adrian.cozma@epfl.ch)

SWIFT Plummer ICs & Orbit Integration Pipeline

This repository contains the complete pipeline to generate initial conditions (ICs) of globular clusters (Plummer models) 
for SWIFT, based on the stellar streams observed by Ibata et al. (2024).

Workflow / Pipeline

Data processing and simulation generation are done in 3 main steps:

Step 1: Observational Data Preparation
File: WorkflowIbata.ipynb
Input: Ibata_catalog.txt
Action: Filters the catalog to keep only the stars that have complete data across the 6 observables (6D phase-space).
Output: stars_6D_Ibata2024.csv

Step 2: Backward Orbit Integration
File: OrbitIntegration_Ibata.ipynb
Input: stars_6D_Ibata2024.csv
Action: Calculates the kinematic median for each stellar stream. Then integrates the position and velocity of 
these medians backward in time for 2 Gyr, using the MWPotential2014 galactic potential (Bovy 2015). 
Orbits ending up too far away are filtered out.
Output: SWIFT_seeds.npy (Positions and velocities at t = -2 Gyr)

Step 3: SWIFT Initial Conditions Generation
Base file: IC_plummer_swift.py
Action: Script containing the physics engine. It relies on the generate_plummer 
function which creates the cluster, defines its mass and radius, and shifts it in 6D space:
generate_plummer(a=0.01, M=1.0e-5, q=1.0, N=10000, bound=2.0,
 boxsize=4.0, use_parallel=True, nb_threads=2, shiftx=0.0, 
 shifty=0.0, shiftz=0.0, shiftvx=0.0, shiftvy=0.0, 
 shiftvz=0.0, fname="plummer.hdf5")

Automation file: auto_multiple_IC_swift.ipynb
Input: SWIFT_seeds.npy
Action: Loops over all the seeds generated in step 2 and calls IC_plummer_swift.py to automatically generate the corresponding Plummer cluster.
Output: initial_conditions/IC_<seed>.hdf5