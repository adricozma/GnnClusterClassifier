from Stellar_objects.cluster_v4_properties import *
from Stellar_objects.visu import *
from Stellar_objects.dispersion_vel_LOS import *
from adri_plotstyle import init_plot_style
from plot_swift_output import load_snapshot

init_plot_style()
import numpy as np
import matplotlib.pyplot as plt


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from galpy.potential import MWPotential2014, evaluateR2derivs, evaluateRzderivs, evaluatez2derivs

path = "Simulations_jed/IC_1_bis/"

outputs_list = [f for f in os.listdir(path) if f.startswith("snapshot_")]
#outputs_list = ["snapshot_0010.hdf5", "snapshot_0100.hdf5"]  # Example: specify the exact snapshot file

#----------------------------------------------------------------------------------------
# We start by using the Cluster class to extract the properties and loop over the outputs 
#----------------------------------------------------------------------------------------
for output in outputs_list: 
     
    fname = os.path.join(path, output)
    pos, vel, _, _ = load_snapshot(fname, shift=500.0)
    cluster = Cluster(pos, vel)
    #cluster.center_cluster()
    # cluster_properties = ClusterProperties(cluster)
    # cluster_properties.plot_VolumicDensity(savefig=False, projection='xy', limit=True, show_orbit=True, orbit_filename="../Simulations/orbit_15h16m05.3s_-00d06m41s.txt",
    #                                        location=f"figures_rapport/VolumicDensity_xy_output_{output:03d}.png")
    
    x, y, z, vx, vy, vz, m = cluster.get_all_particles()

    #-----------------------------------------------------------
    # Start the computation of the Jacobi radius
    #-----------------------------------------------------------
    # Cluster bulk position and velocity (Galactocentric)
    X_cm, Y_cm, Z_cm = np.mean(x), np.mean(y), np.mean(z)
    VX_cm, VY_cm, VZ_cm = np.mean(vx), np.mean(vy), np.mean(vz)
    # as for now we just tke the mean of all particles but could also do a wheighted average (with the total mass)
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    #-----------------------------------------------------------
    # As said in the paper, we ned to be careful about the orientation of the axes 
    # Vectors for the center of mass position and velocity
    R_vec = np.array([X_cm, Y_cm, Z_cm])
    V_vec = np.array([VX_cm, VY_cm, VZ_cm])

    # real angular velocity vector of the cluster (L = R x V)
    L_vec = np.cross(R_vec, V_vec)
    L_norm = np.linalg.norm(L_vec)
    RG = np.linalg.norm(R_vec)
    omega = L_norm / (RG**2)
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # We use an approximatif galactic potential (later should evaluate the potential obtained from simulations at the position of the cluster)
    pot = MWPotential2014
    G = 4.299581e-6 # kpc * (km/s)^2 / Msun

    # Cylindrical coordinate for galpy
    R_cyl = np.sqrt(X_cm**2 + Y_cm**2)
    Z_cyl = Z_cm


    R_vec = np.array([X_cm, Y_cm, Z_cm])
    V_vec = np.array([VX_cm, VY_cm, VZ_cm])

    L_vec = np.cross(R_vec, V_vec)
    L_norm = np.linalg.norm(L_vec)
    R_sph = np.linalg.norm(R_vec) 
    omega = L_norm / (R_sph**2)

    sin_theta = R_cyl / R_sph
    cos_theta = Z_cyl / R_sph

    # cylindrical derivatives (evaluated directly in physical units)
    d2phi_dR2_cyl = evaluateR2derivs(pot, R_cyl, Z_cyl)
    d2phi_dz2     = evaluatez2derivs(pot, R_cyl, Z_cyl)
    d2phi_dRdz    = evaluateRzderivs(pot, R_cyl, Z_cyl)
    # print(d2phi_dR2_cyl)

    # projection of the derivative on the spherical axis (Hessian Matrix projection)
    d2phi_dr2_sph = (sin_theta**2)*d2phi_dR2_cyl + (cos_theta**2)*d2phi_dz2 + 2*sin_theta*cos_theta*d2phi_dRdz

    deno = (omega**2) - d2phi_dr2_sph
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # Coordinates RELATIVE to cluster center (for Jacobi/Profiles)
    dx, dy, dz = x - X_cm, y - Y_cm, z - Z_cm
    dvx, dvy, dvz = vx - VX_cm, vy - VY_cm, vz - VZ_cm
    r_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # Iterative Jacobi Radius (Eq 10, Küpper et al. 2009)
    #-----------------------------------------------------------
    rj = 10  
    for _ in range(20):
        mask_rj = r_3d < rj
        m_cluster = np.sum(m[mask_rj])
        rj_new = np.power((G * m_cluster) / deno, 1/3)
        #rj_new = (G * m_cluster) / deno
        if abs(rj_new - rj) < 1e-4: break # hard coding a limit 
        rj = rj_new
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # base (e_x, e_y, e_z) local
    # e_x towards galactic center (Küpper et al.)
    e_x = - R_vec / R_sph # Changed RG to R_sph as defined above
    # e_z normal to orbital plane
    e_z = L_vec / L_norm
    e_y = np.cross(e_z, e_x)

    # Projection
    pos_rel = np.vstack([dx, dy, dz]) # matrix 3 x N 
    vel_rel = np.vstack([dvx, dvy, dvz])

    dx_rot = np.dot(e_x, pos_rel)
    dy_rot = np.dot(e_y, pos_rel)
    dz_rot = np.dot(e_z, pos_rel)

    dvx_rot = np.dot(e_x, vel_rel)
    dvy_rot = np.dot(e_y, vel_rel)
    dvz_rot = np.dot(e_z, vel_rel)

    # relative kinetic energy
    ekin = 0.5 * (dvx_rot**2 + dvy_rot**2 + dvz_rot**2)
    # potential energy of the cluster
    epot = - (G * m_cluster) / r_3d
    # centrifugal (Corrected sign for tidal energy along x)
    etidal = 0.5 * (omega**2) * (dz_rot**2 - 3 * dx_rot**2)

    e_jacobi = ekin + epot + etidal
    e_crit = - (3 * G * m_cluster) / (2 * rj)
    #-----------------------------------------------------------

    # bounded criteria for each star
    # ----------------------------------------------------------
                                                           #--!?
    is_bound = (e_jacobi < e_crit) & (r_3d < rj)           #--!?
                                                           #--!?
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Final variables for profiles
    r_projected = np.sqrt(dx[is_bound]**2 + dy[is_bound]**2)
    vz_bound = dvz[is_bound]
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # We also have a look at the velocity dispersion and surface density profiles to have a better understanding on how potential escaper influence them
    # Küpper Eqs. 11 - 16

    r_bins = np.logspace(-2, 1, 30)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    sigma_surf, d_sigma_surf = [], []
    v_disp, d_v_disp = [], []

    for i in range(len(r_bins)-1):
        # binning the stars 
        mask = (r_projected >= r_bins[i]) & (r_projected < r_bins[i+1])
        vz_bin = vz_bound[mask]
        n = len(vz_bin)
        area = np.pi * (r_bins[i+1]**2 - r_bins[i]**2)
        
        # Surface Density and its uncertainty
        sigma_surf.append(n / area)  # Convert to stars/pc^2
        d_sigma_surf.append(np.sqrt(n) / area)  # error, also converted to stars/pc^2
        
        # Velocity Dispersion squared and its uncertainty
        if n > 1:
            v_mean = np.mean(vz_bin)
            s2 = np.mean((vz_bin - v_mean)**2)
            # Error on variance Eq. 12 from Küpper et al. 2009
            ds2 = (1/(n*n*s2)) * np.sum(((vz_bin - v_mean)**2 - s2)**2)
            
            v_disp.append(s2**0.5)
            d_v_disp.append(ds2**0.5)
        else:
            v_disp.append(np.nan)
            d_v_disp.append(np.nan)
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # we also need to compoare with the profiles obtained for all stars (including the unbound ones)

    # Projection for all stars
    r_projected_all = np.sqrt(dx**2 + dy**2)

    # compouting surface density for all stars

    sigma_surf_all, d_sigma_surf_all = [], []
    for i in range(len(r_bins)-1):
    
        mask_all = (r_projected_all >= r_bins[i]) & (r_projected_all < r_bins[i+1])
        n_all = np.sum(mask_all)
        area = np.pi * (r_bins[i+1]**2 - r_bins[i]**2)
        
        sigma_surf_all.append(n_all / area)
        d_sigma_surf_all.append(np.sqrt(n_all) / (area))

    # Computing the velocity dispersion for all stars 

    v_disp_all, d_v_disp_all = [], []

    for i in range(len(r_bins)-1):
        mask_all = (r_projected_all >= r_bins[i]) & (r_projected_all < r_bins[i+1])
        vz_bin_all = dvz[mask_all]
        n_all = len(vz_bin_all)
        
        if n_all > 1:
            v_mean_all = np.mean(vz_bin_all)
            s2_all = np.mean((vz_bin_all - v_mean_all)**2)
            # Error calculation
            ds2_all = (1/(n_all*n_all*s2_all)) * np.sum(((vz_bin_all - v_mean_all)**2 - s2_all)**2)

            # Taking the square root to get sigma 
            v_disp_all.append(np.sqrt(s2_all))

            d_v_disp_all.append(ds2_all ** 0.5)
        else:
            v_disp_all.append(np.nan)
            d_v_disp_all.append(np.nan)
    # folder = "bound_matrices/Kepler_17kpc/"
    # filename = f"{folder}bound_mask_output_live_{output}.npz"
    # np.savez_compressed(filename, is_bound=is_bound.astype(bool))
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
#    # Final Plotting 
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
#    # plt.subplots_adjust(hspace=0.1)
   
#     # Top Plot: Surface Density
#     ax1.errorbar(r_centers, sigma_surf, yerr=d_sigma_surf, fmt='x', color='royalblue', 
#                 ecolor='gray', markersize=4, capsize=2, label=r'$\mathrm{Bound\ stars\ \Sigma(R)}$')
#     ax1.errorbar(r_centers, sigma_surf_all, yerr=d_sigma_surf_all, fmt='--', color='gray', 
#                 alpha=0.6, label=r'$\Sigma(R)\ \mathrm{All\ stars}$')
#     ax1.set_yscale('log')
#     ax1.set_xscale('log')
#     ax1.set_ylabel(r'$\Sigma\ \mathrm{[stars \cdot kpc^{-2}]}$')
#     ax1.grid(True, which="both", ls="-", alpha=0.7)
#     ax1.legend()

#     # Bottom Plot: Velocity Dispersion
#     ax2.errorbar(r_centers, v_disp, yerr=d_v_disp, fmt='x', color='crimson', 
#                 ecolor='gray', markersize=4, capsize=2, label=r'$\mathrm{Bound\ stars\ \sigma_{LOS}}$')
#     ax2.scatter(r_centers, v_disp_all, marker='x', color='gray', 
#                 alpha=0.5, label=r'$\mathrm{All\ stars\ (including\ tails)}$')
#     ax2.set_xscale('log')
#     ax2.set_yscale('log')
#     ax2.set_ylim(0.1, 10)
#     ax2.set_ylabel(r'$\sigma_{\mathrm{LOS}}\ \mathrm{[km \cdot s^{-1}]}$')
#     ax2.set_xlabel(r'$R\ \mathrm{[kpc]}$')
#     ax2.grid(True, which="both", ls="-", alpha=0.7)
#     ax2.legend()
#     # plt.savefig('../Simulations/point_mass_orbit_17kpc/figures/bound_plots/cluster_observables_improved_output={}.png'.format(output), dpi=300)
#     plt.suptitle('Cluster Observables (Bound Stars Only)', fontsize=14)


# trying to make a scatter plot of stars that are actually bounded to the cluster and the ones that are nont

    plt.figure(figsize=(8, 8))
    plt.grid(True, which="both", ls="-", alpha=0.7, zorder = 1)
    plt.title(' Bound vs Unbound Stars', fontsize=14)
    plt.scatter(dx[~is_bound], dy[~is_bound], color='lightgray', s=10, label=r'$\mathrm{Unbound\ Stars}$')
    plt.scatter(dx[is_bound], dy[is_bound], color='blue', s=10, label=r'$\mathrm{Bound\ Stars}$')
    plt.xlabel(r'$X\ \mathrm{[kpc]}$')
    plt.ylabel(r'$Y\ \mathrm{[kpc]}$')
    plt.legend()
    plt.savefig('Simulations_jed/IC_1_bis/figures/bound/bound_vs_unbound_stars_output={}.png'.format(output.split(".")[0]), dpi=300)