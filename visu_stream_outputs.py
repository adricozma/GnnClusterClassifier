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
path = "outputs_SWIFT_jed/withoutDF/IC_54/"
path = "/Volumes/Adrian 1 SS/MA2/TP4b/outputs_SWIFT_jed/withoutDF/IC_54"
save_folder = "Plots/IC_54_orbit/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

output_list = [f for f in os.listdir(path) if f.startswith("snapshot_")]
#output_list = ["snapshot_0050.hdf5", "snapshot_0060.hdf5", "snapshot_0070.hdf5", "snapshot_0080.hdf5"]  # Example: specify the exact snapshot file
orbit_file = "make_ICs/orbits/orbit_54.txt"
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
