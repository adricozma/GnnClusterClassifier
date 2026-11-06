import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from Stellar_objects.cluster_v4_properties import *
from Stellar_objects.visu import *
from Stellar_objects.dispersion_vel_LOS import *

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, ICRS

def get_observables_matrix(pos, vel):
    # pos: array [N, 3] en kpc
    # vel: array [N, 3] en km/s
    
    sim_coords = SkyCoord(
        x=pos[:, 0] * u.kpc,
        y=pos[:, 1] * u.kpc,
        z=pos[:, 2] * u.kpc,
        v_x=vel[:, 0] * (u.km / u.s),
        v_y=vel[:, 1] * (u.km / u.s),
        v_z=vel[:, 2] * (u.km / u.s),
        frame=Galactocentric()
    )

    obs_coords = sim_coords.transform_to(ICRS())

    return np.column_stack((
        obs_coords.ra.degree, #ok
        obs_coords.dec.degree, #ok
        obs_coords.distance.kpc, #not sure 
        obs_coords.pm_ra_cosdec.value, #ok
        obs_coords.pm_dec.value, #ok
        obs_coords.radial_velocity.value #v_los? 
    ))


from scipy.spatial.transform import Rotation as R

def plot_observables(matrix_obs, output_name="rotation_check.png"):
    ra = matrix_obs[:, 0]
    dec = matrix_obs[:, 1]
    v_los = matrix_obs[:, 5]

    sc = plt.scatter(ra, dec, c=v_los, s=1, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='v_{los} [km/s]')
    
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    plt.title(f'observed projection (random rotation)')
    plt.grid(True, alpha=0.2)
    

    plt.gca().invert_xaxis() 
    
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()

def rotate_data(pos, vel):
    matrix = R.from_euler('xyz', np.random.uniform(0, 2*np.pi, 3)).as_matrix()
    return np.dot(pos, matrix.T), np.dot(vel, matrix.T)


def standardise_matrix(matrix, mode="physical", ref_scales=(2.0, 2.0, 1.5, 1.0, 1.0, 15.0)):
    """
    Normalizes the 6D observables matrix [RA (deg), Dec (deg), D (kpc), pm_ra_cosdec (mas/yr), pm_dec (mas/yr), v_los (km/s)].
    
    Parameters:
    - mode:
        * 'physical' (recommended): Centers coordinates on the stream median (with circular wrapping and cos(Dec)
          correction for RA) and divides by fixed physical reference scales `ref_scales`. This preserves the true
          physical compactness of bound cores vs. dissolved tidal streams across snapshots.
        * 'zscore' (legacy): Per-snapshot z-score standardization (matrix - mean) / std.
    - ref_scales: tuple of characteristic physical scales (deg, deg, kpc, mas/yr, mas/yr, km/s).
    """
    matrix = np.asarray(matrix, dtype=float)
    if mode == "zscore":
        means = np.mean(matrix, axis=0)
        stds = np.where(np.std(matrix, axis=0) > 1e-8, np.std(matrix, axis=0), 1.0)
        return (matrix - means) / stds, means, stds

    # Circular median/mean for RA to avoid 0/360 deg wrap-around discontinuities
    ra_rad = np.deg2rad(matrix[:, 0])
    ra_center_rad = np.arctan2(np.median(np.sin(ra_rad)), np.median(np.cos(ra_rad)))
    ra_center_deg = np.rad2deg(ra_center_rad) % 360.0

    centers = np.median(matrix, axis=0)
    centers[0] = ra_center_deg

    centered = matrix - centers
    # Wrap dRA into [-180, 180] deg and project onto local tangent sky plane via cos(Dec_center)
    centered[:, 0] = ((matrix[:, 0] - ra_center_deg + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(centers[1]))

    scales = np.asarray(ref_scales[: matrix.shape[1]], dtype=float)
    return centered / scales, centers, scales



# path = "../Simulations/point_mass_orbit_17kpc/"

# outputs_list = list(range(1,202))  # Outputs 

# path_folder = "outputs_rot_convunits/kepler_17kpc/"
# for output in outputs_list:
#     final_matrices = []
#     cluster = Cluster(path, output)
#     cluster.remove_ParentGalaxy(Mmax=1e3)

    
#     pos_raw = np.column_stack((cluster.x, cluster.y, getattr(cluster, 'z', np.zeros_like(cluster.x))))
#     vel_raw = np.column_stack((cluster.vx, cluster.vy, getattr(cluster, 'vz', np.zeros_like(cluster.vx))))

#     n = 5
#     for _ in range(n): # make n different rotations for augmenting the data
#         pos_rot, vel_rot = rotate_data(pos_raw, vel_raw)
#         matrix_obs = get_observables_matrix(pos_rot, vel_rot)

#         #simple quick plot to check the rotations are working as expected (should be different for each rotation)
#         # if output == outputs_list[0] and n == 1:
#         #     plot_observables(matrix_obs, f"obs_output_{output}.png")


#         X_ready, mu, sigma = standardise_matrix(matrix_obs)
#         final_matrices.append(X_ready)
#     # saving the final matrices for each output and rotation
#     final_matrices = np.array(final_matrices) # shape (n, N, 6)
#     np.save(f"{path_folder}final_matrices_output_{output}", final_matrices)
    
