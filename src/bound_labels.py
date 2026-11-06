from galpy.potential import MWPotential2014, evaluateR2derivs, evaluateRzderivs, evaluatez2derivs
from scipy.spatial import cKDTree
import numpy as np


def compute_cluster_center_density(x, y, z, vx, vy, vz, m, k=32, n_core=100):
    """
    Finds the progenitor cluster center (X_cm, Y_cm, Z_cm) and bulk velocity (VX_cm, VY_cm, VZ_cm)
    using 3D k-NN local density estimation (Casertano & Hut 1985) followed by shrinking-sphere
    refinement. This prevents extended or curved tidal tails from pulling the center out of the core.
    """
    n_total = len(x)
    if n_total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pos = np.column_stack((x, y, z))
    vel = np.column_stack((vx, vy, vz))

    k_eff = min(k, n_total - 1)
    if k_eff < 1:
        w = m / np.sum(m)
        c_pos = np.sum(pos * w[:, None], axis=0)
        c_vel = np.sum(vel * w[:, None], axis=0)
        return c_pos[0], c_pos[1], c_pos[2], c_vel[0], c_vel[1], c_vel[2]

    tree = cKDTree(pos)
    distances, _ = tree.query(pos, k=k_eff + 1)
    r_k = np.maximum(distances[:, -1], 1e-6)
    rho = m * (k_eff / ((4.0 / 3.0) * np.pi * (r_k ** 3)))

    # Initial guess: density-weighted center around the densest particle
    peak_idx = int(np.argmax(rho))
    c_pos = pos[peak_idx].copy()

    # Shrinking spheres refinement around the density peak (from 0.5 kpc down to 0.05 kpc)
    for r_sphere in (0.5, 0.2, 0.1, 0.05):
        d2 = np.sum((pos - c_pos) ** 2, axis=1)
        mask = d2 < (r_sphere ** 2)
        if np.sum(mask) >= max(20, min(n_core, n_total // 20)):
            w = rho[mask] * m[mask]
            w_sum = np.sum(w)
            if w_sum > 0:
                c_pos = np.sum(pos[mask] * (w / w_sum)[:, None], axis=0)

    # Core velocity: mass-weighted velocity of the n_core stars closest to c_pos
    n_vel = min(n_core, n_total)
    _, core_idx = tree.query(c_pos, k=n_vel)
    core_idx = np.atleast_1d(core_idx)
    w_core = m[core_idx]
    w_core_sum = np.sum(w_core)
    if w_core_sum > 0:
        c_pos = np.sum(pos[core_idx] * (w_core / w_core_sum)[:, None], axis=0)
        c_vel = np.sum(vel[core_idx] * (w_core / w_core_sum)[:, None], axis=0)
    else:
        c_vel = np.mean(vel[core_idx], axis=0)

    return float(c_pos[0]), float(c_pos[1]), float(c_pos[2]), float(c_vel[0]), float(c_vel[1]), float(c_vel[2])


def compute_bound_stars(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    m,
    G=4.299581e-6,
    potential=MWPotential2014,
    centering="density",
    ro=8.0,
    vo=220.0,
    return_potential_escapers=False,
):
    # We follow the method described in Küpper et al. (2010a,b) to compute the Jacobi radius and identify bound stars
    """
    Input: arrays of positions (x, y, z) in kpc and velocities (vx, vy, vz) in km/s of stars in the cluster,
    particle masses m in M_sun, G in kpc*(km/s)^2/M_sun, and the galactic potential (default: MWPotential2014).
    - centering: 'density' (recommended, 3D density peak + shrinking spheres) or 'mean' (legacy arithmetic mean).
    - return_potential_escapers: if True, also returns is_potential_escaper (r_3d < rj and e_jacobi >= e_crit).
    Output: returns a boolean array indicating which stars are bound to the cluster
    and the profiles (surface density and velocity dispersion) for the bound stars and for all stars (for comparison).
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)
    m = np.asarray(m, dtype=float)

    #-----------------------------------------------------------
    # Start the computation of the Jacobi radius
    #-----------------------------------------------------------
    # Cluster bulk position and velocity (Galactocentric)
    if centering == "density":
        X_cm, Y_cm, Z_cm, VX_cm, VY_cm, VZ_cm = compute_cluster_center_density(x, y, z, vx, vy, vz, m)
    else:
        X_cm, Y_cm, Z_cm = np.mean(x), np.mean(y), np.mean(z)
        VX_cm, VY_cm, VZ_cm = np.mean(vx), np.mean(vy), np.mean(vz)
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # Vectors for the center of mass position and velocity
    R_vec = np.array([X_cm, Y_cm, Z_cm])
    V_vec = np.array([VX_cm, VY_cm, VZ_cm])

    # Angular velocity vector of the cluster (L = R x V)
    L_vec = np.cross(R_vec, V_vec)
    L_norm = np.linalg.norm(L_vec)
    R_sph = max(np.linalg.norm(R_vec), 1e-8)
    omega = L_norm / (R_sph ** 2)
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    pot = potential

    # Cylindrical coordinates converted to galpy natural units (R/ro, Z/ro)
    # and scaled back to physical units (km/s/kpc)^2 via (vo/ro)^2
    R_cyl = np.sqrt(X_cm ** 2 + Y_cm ** 2)
    Z_cyl = Z_cm

    sin_theta = R_cyl / R_sph
    cos_theta = Z_cyl / R_sph

    unit_conv = (vo / ro) ** 2
    d2phi_dR2_cyl = float(evaluateR2derivs(pot, R_cyl / ro, Z_cyl / ro)) * unit_conv
    d2phi_dz2     = float(evaluatez2derivs(pot, R_cyl / ro, Z_cyl / ro)) * unit_conv
    d2phi_dRdz    = float(evaluateRzderivs(pot, R_cyl / ro, Z_cyl / ro)) * unit_conv

    # Projection of the derivative on the spherical radial axis (Hessian matrix projection)
    d2phi_dr2_sph = (sin_theta ** 2) * d2phi_dR2_cyl + (cos_theta ** 2) * d2phi_dz2 + 2.0 * sin_theta * cos_theta * d2phi_dRdz

    deno = (omega ** 2) - d2phi_dr2_sph
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # Coordinates RELATIVE to cluster center (for Jacobi/Profiles)
    dx, dy, dz = x - X_cm, y - Y_cm, z - Z_cm
    dvx, dvy, dvz = vx - VX_cm, vy - VY_cm, vz - VZ_cm
    r_3d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # Iterative Jacobi Radius (Eq. 12, Küpper et al. 2010b)
    #-----------------------------------------------------------
    rj = 0.5  # Initial guess in kpc around the progenitor density center
    m_cluster = 0.0
    if deno > 1e-12:
        for _ in range(20):
            mask_rj = r_3d < rj
            m_cluster = float(np.sum(m[mask_rj]))
            if m_cluster <= 0.0:
                rj = 0.0
                break
            rj_new = float(np.power((G * m_cluster) / deno, 1.0 / 3.0))
            if abs(rj_new - rj) < 1e-4:
                rj = rj_new
                break
            rj = rj_new
        mask_rj = r_3d < rj
        m_cluster = float(np.sum(m[mask_rj]))
    else:
        rj = 0.0
        m_cluster = 0.0
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    # Local co-rotating basis (e_x, e_y, e_z)
    # e_x towards galactic center (Küpper et al.)
    e_x = -R_vec / R_sph
    # e_z normal to orbital plane
    e_z = L_vec / max(L_norm, 1e-8)
    e_y = np.cross(e_z, e_x)

    # Projection
    pos_rel = np.vstack([dx, dy, dz])  # matrix 3 x N
    vel_rel = np.vstack([dvx, dvy, dvz])

    dx_rot = np.dot(e_x, pos_rel)
    dy_rot = np.dot(e_y, pos_rel)
    dz_rot = np.dot(e_z, pos_rel)

    dvx_rot = np.dot(e_x, vel_rel)
    dvy_rot = np.dot(e_y, vel_rel)
    dvz_rot = np.dot(e_z, vel_rel)

    if rj > 0.0 and m_cluster > 0.0:
        # Relative kinetic energy per unit mass
        ekin = 0.5 * (dvx_rot ** 2 + dvy_rot ** 2 + dvz_rot ** 2)
        # Potential energy of the cluster (point-mass approximation outside core)
        epot = -(G * m_cluster) / np.maximum(r_3d, 1e-8)
        # Centrifugal and tidal potential term
        etidal = 0.5 * (omega ** 2) * (dz_rot ** 2 - 3.0 * dx_rot ** 2)

        e_jacobi = ekin + epot + etidal
        e_crit = -(3.0 * G * m_cluster) / (2.0 * rj)

        is_bound = (e_jacobi < e_crit) & (r_3d < rj)
        is_potential_escaper = (e_jacobi >= e_crit) & (r_3d < rj)
    else:
        is_bound = np.zeros(len(x), dtype=bool)
        is_potential_escaper = np.zeros(len(x), dtype=bool)
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

    if return_potential_escapers:
        return (
            is_bound,
            is_potential_escaper,
            r_centers,
            sigma_surf,
            d_sigma_surf,
            v_disp,
            d_v_disp,
            sigma_surf_all,
            d_sigma_surf_all,
            v_disp_all,
            d_v_disp_all,
        )

    return is_bound, r_centers, sigma_surf, d_sigma_surf, v_disp, d_v_disp, sigma_surf_all, d_sigma_surf_all, v_disp_all, d_v_disp_all