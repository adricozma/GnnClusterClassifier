import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Stellar_objects.cluster_v4_properties import *
from Stellar_objects.visu import *
from Stellar_objects.dispersion_vel_LOS import *
from plot_swift_output import load_snapshot

from bound_labels import compute_bound_stars

from adri_plotstyle import init_plot_style
init_plot_style()
import numpy as np
import matplotlib.pyplot as plt

#%%
path = os.path.join(ROOT_DIR, "outputs_SWIFT_jed/withoutDF/IC_54/")
save_folder = os.path.join(ROOT_DIR, "plots/IC_54_orbit/")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

output_list = [f for f in os.listdir(path) if f.startswith("snapshot_")]
orbit_file = os.path.join(ROOT_DIR, "make_ICs/orbits/orbit_54.txt")
for output in output_list:
    fname = os.path.join(path, output)
    pos, vel, _, _ = load_snapshot(fname, shift=500.0)
    is_bound, _, _, _, _, _, _, _, _, _ = compute_bound_stars(pos[:,0], pos[:,1], pos[:,2], vel[:,0], vel[:,1], vel[:,2], np.ones_like(pos[:,0]))
    cluster = Cluster(pos=pos, vel=vel, Ramses = False)
    properties = ClusterProperties(cluster)
    properties.plot_VolumicDensity(savefig =True, location=save_folder, fname = output.split(".")[0], limit=True, axlim=10, show_orbit=True, orbit_filename=orbit_file)
    
    # plt.figure(figsize=(8, 8))
    # plt.grid(True, which="both", ls="-", alpha=0.7, zorder = 1)
    # plt.title(' $\mathrm{Bound\ vs\ Unbound\ Stars}$', fontsize=14)
    # plt.scatter(pos[:,0][~is_bound], pos[:,1][~is_bound], color='lightgray', s=10, label=r'$\mathrm{Unbound\ Stars}$')
    # plt.scatter(pos[:,0][is_bound], pos[:,1][is_bound], color='blue', s=10, label=r'$\mathrm{Bound\ Stars}$')
    # plt.xlabel(r'$X\ \mathrm{[kpc]}$')
    # plt.ylabel(r'$Y\ \mathrm{[kpc]}$')
    # plt.axis("equal")
    # plt.legend()

# %%
