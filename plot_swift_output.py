import numpy as np
# from matplotlib import pyplot as plt
# import argparse
# from scipy.optimize import curve_fit
import h5py

# # Parse user input
# parser = argparse.ArgumentParser(
#     description="Plot multiple density profiles against theoretical prediction"
# )
# parser.add_argument("files", nargs="+", help="snapshot files to be imaged")
# parser.add_argument("--notex", action="store_true", help="Flag to not use LaTeX markup")
# parser.add_argument(
#     "-a", type=float, default=0.1, help="Softening length of theoretical model"
# )
# parser.add_argument(
#     "-M", type=float, default=1.0e-5, help="Total Mass of theoretical model"
# )
# parser.add_argument("-Rmin", type=float, default=0.01, help="Min Radius")
# parser.add_argument("-Rmax", type=float, default=2.1, help="Max Radius")
# parser.add_argument(
#     "-nbsamples", type=int, default=64, help="Number of radii to sample (bins)"
# )
# parser.add_argument(
#     "-shift", type=float, default=2.0, help="Shift applied to particles in params.yml"
# )

# args = parser.parse_args()
# fnames = args.files

# # Limit number of snapshots to plot
# if len(fnames) > 20:
#     raise ValueError(
#         "Too many ({:d}) files provided (cannot plot more than 20).".format(len(fnames))
#     )

# # Set parameters
# tex = not args.notex
# if tex:
#     plt.rcParams.update({"text.usetex": tex, "font.size": 14, "font.family": "serif"})
# else:
#     plt.rcParams.update({"font.size": 12})
# figsize = 7

# # # Model Parameters (Mestel surface density)
# # G = 4.299581e06
# # rsp = np.logspace(np.log10(args.Rmin), np.log10(args.Rmax), args.nbsamples)

# #plotting the particles positions 
# fig, ax = plt.subplots(figsize=(1.2 * figsize, figsize))
# for fname in fnames:    
#     print(fname)
#     f = h5py.File(fname, "r")
#     pos = np.array(f["DMParticles"]["Coordinates"]) - args.shift
#     time = (
#         f["Header"].attrs["Time"][0]
#         * f["Units"].attrs["Unit time in cgs (U_t)"][0]
#         / 31557600.0e9
#     )
#     mass = np.array(f["DMParticles"]["Masses"])
#     x = pos[:, 0]
#     y = pos[:, 1]
#     z = pos[:, 2]
#     ax.plot(y,z, "o", ms=1.7, label=r"$t=$ {:.3f} Gyr".format(time))
# ax.set_xlabel("y [kpc]")
# ax.set_ylabel("z [kpc]")
# ax.set_aspect("equal")
# # ax.set_xlim(-50, 50)
# # ax.set_ylim(-50, 50)
# ax.legend()

# ax.set_title(
#     r"Plummer Particle Positions"
# )
# plt.tight_layout()

# plt.savefig("Simulations_jed/IC_1_bis/particle_positions.png")

def load_snapshot(fname, shift):
    # Load snapshot data from a given file and apply a shift to the particle positions.
    # returns particle positions, masses, and time of the snapshot
    f = h5py.File(fname, "r")
    pos = np.array(f["DMParticles"]["Coordinates"]) - shift
    vel = np.array(f["DMParticles"]["Velocities"])

    time = (
        f["Header"].attrs["Time"][0]
        * f["Units"].attrs["Unit time in cgs (U_t)"][0]
        / 31557600.0e9
    )
    mass = np.array(f["DMParticles"]["Masses"])
    return pos, vel, mass, time