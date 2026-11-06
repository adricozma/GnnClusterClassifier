from galpy.potential import MWPotential2014, evaluateR2derivs, evaluateRzderivs, evaluatez2derivs
from scipy.spatial import cKDTree
import numpy as np


def compute_cluster_center_density(x, y, z, vx, vy, vz, m, k=32, n_core=100, tol=1e-5):
    """
    Finds the progenitor cluster center (X_cm, Y_cm, Z_cm) and bulk velocity (VX_cm, VY_cm, VZ_cm)
    using 3D k-NN local density estimation (Casertano & Hut 1985) followed by shrinking-sphere
    and smooth Gaussian-kernel mean-shift refinement. This prevents extended or curved tidal tails
    from pulling the center out of the surviving core and avoids discrete boundary oscillations.
    """
    n_total = len(x)
    if n_total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([])

    pos = np.column_stack((x, y, z))
    vel = np.column_stack((vx, vy, vz))

    k_eff = min(k, n_total - 1)
    if k_eff < 1:
        w = m / np.sum(m)
        c_pos = np.sum(pos * w[:, None], axis=0)
        c_vel = np.sum(vel * w[:, None], axis=0)
        return float(c_pos[0]), float(c_pos[1]), float(c_pos[2]), float(c_vel[0]), float(c_vel[1]), float(c_vel[2]), np.ones(n_total)

    tree = cKDTree(pos)
    distances, _ = tree.query(pos, k=k_eff + 1)
    r_k = np.maximum(distances[:, -1], 1e-6)
    rho = m * (k_eff / ((4.0 / 3.0) * np.pi * (r_k ** 3)))

    # Initial guess: densest particle
    peak_idx = int(np.argmax(rho))
    c_pos = pos[peak_idx].copy()

    # Coarse shrinking spheres around the density peak (from 0.5 kpc down to 0.05 kpc)
    for r_sphere in (0.5, 0.2, 0.1, 0.05):
        d2 = np.sum((pos - c_pos) ** 2, axis=1)
        mask = d2 < (r_sphere ** 2)
        if np.sum(mask) >= max(20, min(n_core, n_total // 20)):
            w = rho[mask] * m[mask]
            w_sum = np.sum(w)
            if w_sum > 0:
                c_pos = np.sum(pos[mask] * (w / w_sum)[:, None], axis=0)

    # Smooth Gaussian mean-shift refinement around the core (eliminates discrete top-hat boundary jumps)
    n_vel = min(max(n_core, 50), n_total)
    dists_core, _ = tree.query(c_pos, k=n_vel)
    h_kernel = max(float(np.max(dists_core)), 0.005)
    for _ in range(15):
        d2 = np.sum((pos - c_pos) ** 2, axis=1)
        mask = d2 < (9.0 * h_kernel ** 2)
        if np.sum(mask) < 10:
            break
        w = rho[mask] * m[mask] * np.exp(-0.5 * d2[mask] / (h_kernel ** 2))
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            break
        c_next = np.sum(pos[mask] * (w / w_sum)[:, None], axis=0)
        shift = float(np.linalg.norm(c_next - c_pos))
        c_pos = c_next
        if shift < tol:
            break

    # Core velocity: smoothly weighted by rho * m * Gaussian kernel around converged c_pos
    d2 = np.sum((pos - c_pos) ** 2, axis=1)
    mask = d2 < (4.0 * h_kernel ** 2)
    if np.sum(mask) >= 5:
        w_core = rho[mask] * m[mask] * np.exp(-0.5 * d2[mask] / (h_kernel ** 2))
        w_core_sum = float(np.sum(w_core))
        c_vel = np.sum(vel[mask] * (w_core / w_core_sum)[:, None], axis=0)
    else:
        _, core_idx = tree.query(c_pos, k=min(n_core, n_total))
        core_idx = np.atleast_1d(core_idx)
        c_vel = np.mean(vel[core_idx], axis=0)

    return float(c_pos[0]), float(c_pos[1]), float(c_pos[2]), float(c_vel[0]), float(c_vel[1]), float(c_vel[2]), rho


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
    softening=0.001,
    min_stars=15,
    max_iter=30,
    rtol=1e-5,
    return_potential_escapers=False,
    return_diagnostics=False,
):
    """
    Computes the Jacobi radius R_J and identifies gravitationally bound stars and potential escapers
    following Küpper et al. (2010a,b) and Binney & Tremaine (2008).

    Algorithmic improvements over naive fixed-point iteration:
    1. 3D k-NN local density centering (Casertano & Hut 1985) + shrinking spheres + smooth Gaussian
       mean-shift prevents extended tidal tails from pulling the center of mass outside the surviving
       core and eliminates discrete boundary oscillations.
    2. Exact monotonic root solver on sorted radial shells: finds the outermost radius where the
       enclosed mean density exceeds the critical tidal density, r_(k) < R_J(k) = (G * M_k / deno)^(1/3),
       coupled with damped fixed-point verification until N(<R_J) and R_J converge within rtol.
       Eliminates dependence on arbitrary initial R_J guesses and 2-cycle oscillations.
    3. Physically consistent effective tidal potential (-0.5 * deno * x_rot^2 + 0.5 * omega^2 * z_rot^2)
       and Plummer-softened cluster potential (-G * M_c / sqrt(r^2 + softening^2)) so that E_J(+-R_J, 0, 0)
       matches E_crit identically at the L1/L2 Lagrange points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)
    m = np.asarray(m, dtype=float)
    n_total = len(x)

    # 1. Cluster core position and velocity (Galactocentric)
    if centering == "density":
        X_cm, Y_cm, Z_cm, VX_cm, VY_cm, VZ_cm, _ = compute_cluster_center_density(x, y, z, vx, vy, vz, m)
    else:
        X_cm, Y_cm, Z_cm = float(np.mean(x)), float(np.mean(y)), float(np.mean(z))
        VX_cm, VY_cm, VZ_cm = float(np.mean(vx)), float(np.mean(vy)), float(np.mean(vz))

    # 2. Galactocentric orbital angular velocity and host potential second derivatives
    R_vec = np.array([X_cm, Y_cm, Z_cm])
    V_vec = np.array([VX_cm, VY_cm, VZ_cm])

    L_vec = np.cross(R_vec, V_vec)
    L_norm = float(np.linalg.norm(L_vec))
    R_sph = max(float(np.linalg.norm(R_vec)), 1e-8)
    omega = L_norm / (R_sph ** 2)

    R_cyl = float(np.sqrt(X_cm ** 2 + Y_cm ** 2))
    Z_cyl = float(Z_cm)
    sin_theta = R_cyl / R_sph
    cos_theta = Z_cyl / R_sph

    unit_conv = (vo / ro) ** 2
    d2phi_dR2_cyl = float(evaluateR2derivs(potential, R_cyl / ro, Z_cyl / ro)) * unit_conv
    d2phi_dz2     = float(evaluatez2derivs(potential, R_cyl / ro, Z_cyl / ro)) * unit_conv
    d2phi_dRdz    = float(evaluateRzderivs(potential, R_cyl / ro, Z_cyl / ro)) * unit_conv

    d2phi_dr2_sph = (
        (sin_theta ** 2) * d2phi_dR2_cyl
        + (cos_theta ** 2) * d2phi_dz2
        + 2.0 * sin_theta * cos_theta * d2phi_dRdz
    )
    deno = (omega ** 2) - d2phi_dr2_sph

    # Relative coordinates from converged core center
    dx, dy, dz = x - X_cm, y - Y_cm, z - Z_cm
    dvx, dvy, dvz = vx - VX_cm, vy - VY_cm, vz - VZ_cm
    r_3d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # 3. Exact Monotonic Tidal Root + Self-Consistent Convergence for R_J and M(<R_J)
    rj = 0.0
    m_cluster = 0.0
    converged = False
    n_iter = 0

    if deno <= 1e-12:
        converged = True
    else:
        order = np.argsort(r_3d)
        r_sorted = r_3d[order]
        m_cum = np.cumsum(m[order])
        rj_candidates = np.power((G * m_cum) / deno, 1.0 / 3.0)

        # Find all shells k >= min_stars where mean enclosed density exceeds critical tidal density (R_J(k) > r_(k))
        valid = np.where((np.arange(n_total) >= min_stars - 1) & (rj_candidates > r_sorted))[0]
        if len(valid) == 0:
            rj = 0.0
            m_cluster = 0.0
            converged = True
            n_iter = 1
        else:
            # Start from the outermost crossing and iterate to exact discrete fixed point M(<R_J)
            idx_star = int(valid[-1])
            rj = float(rj_candidates[idx_star])
            m_cluster = float(m_cum[idx_star])

            for it in range(1, max_iter + 1):
                n_iter = it
                # Binary search on sorted radii for exact O(log N) enclosed mass M(<rj)
                k_enc = int(np.searchsorted(r_sorted, rj, side="left"))
                if k_enc < min_stars:
                    rj = 0.0
                    m_cluster = 0.0
                    converged = True
                    break

                m_new = float(m_cum[k_enc - 1])
                rj_new = float(np.power((G * m_new) / deno, 1.0 / 3.0))
                k_next = int(np.searchsorted(r_sorted, rj_new, side="left"))

                rel_diff = abs(rj_new - rj) / max(rj_new, 1e-8)
                if k_next == k_enc or rel_diff < rtol:
                    rj = rj_new
                    m_cluster = m_new
                    converged = True
                    break

                # If discrete step jumps back and forth, damp the radius update
                if it > 5:
                    rj = 0.5 * (rj + rj_new)
                else:
                    rj = rj_new
                m_cluster = m_new

    # 4. Co-rotating frame projection and Jacobi energy E_J (Küpper et al. 2010a,b)
    e_x = -R_vec / R_sph
    e_z = L_vec / max(L_norm, 1e-8)
    e_y = np.cross(e_z, e_x)

    pos_rel = np.vstack([dx, dy, dz])
    vel_rel = np.vstack([dvx, dvy, dvz])

    dx_rot = np.dot(e_x, pos_rel)
    dy_rot = np.dot(e_y, pos_rel)
    dz_rot = np.dot(e_z, pos_rel)

    dvx_rot = np.dot(e_x, vel_rel)
    dvy_rot = np.dot(e_y, vel_rel)
    dvz_rot = np.dot(e_z, vel_rel)

    if rj > 0.0 and m_cluster > 0.0:
        ekin = 0.5 * (dvx_rot ** 2 + dvy_rot ** 2 + dvz_rot ** 2)
        epot = -(G * m_cluster) / np.sqrt(r_3d ** 2 + softening ** 2)
        etidal = -0.5 * deno * (dx_rot ** 2) + 0.5 * (omega ** 2) * (dz_rot ** 2)

        e_jacobi = ekin + epot + etidal
        e_crit = -(G * m_cluster) / np.sqrt(rj ** 2 + softening ** 2) - 0.5 * deno * (rj ** 2)

        is_bound = (e_jacobi < e_crit) & (r_3d < rj)
        is_potential_escaper = (e_jacobi >= e_crit) & (r_3d < rj)
    else:
        is_bound = np.zeros(n_total, dtype=bool)
        is_potential_escaper = np.zeros(n_total, dtype=bool)

    # 5. Surface density and velocity dispersion profiles (Küpper et al. 2010a Eqs. 11-16)
    r_projected = np.sqrt(dx[is_bound] ** 2 + dy[is_bound] ** 2)
    vz_bound = dvz[is_bound]

    r_bins = np.logspace(-2, 1, 30)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    sigma_surf, d_sigma_surf = [], []
    v_disp, d_v_disp = [], []

    for i in range(len(r_bins) - 1):
        mask = (r_projected >= r_bins[i]) & (r_projected < r_bins[i + 1])
        vz_bin = vz_bound[mask]
        n = len(vz_bin)
        area = np.pi * (r_bins[i + 1] ** 2 - r_bins[i] ** 2)

        sigma_surf.append(n / area)
        d_sigma_surf.append(np.sqrt(n) / area)

        if n > 1:
            v_mean = np.mean(vz_bin)
            s2 = np.mean((vz_bin - v_mean) ** 2)
            ds2 = (1.0 / (n * n * max(s2, 1e-12))) * np.sum(((vz_bin - v_mean) ** 2 - s2) ** 2)
            v_disp.append(s2 ** 0.5)
            d_v_disp.append(ds2 ** 0.5)
        else:
            v_disp.append(np.nan)
            d_v_disp.append(np.nan)

    r_projected_all = np.sqrt(dx ** 2 + dy ** 2)
    sigma_surf_all, d_sigma_surf_all = [], []
    v_disp_all, d_v_disp_all = [], []

    for i in range(len(r_bins) - 1):
        mask_all = (r_projected_all >= r_bins[i]) & (r_projected_all < r_bins[i + 1])
        n_all = int(np.sum(mask_all))
        area = np.pi * (r_bins[i + 1] ** 2 - r_bins[i] ** 2)

        sigma_surf_all.append(n_all / area)
        d_sigma_surf_all.append(np.sqrt(n_all) / area)

        vz_bin_all = dvz[mask_all]
        if n_all > 1:
            v_mean_all = np.mean(vz_bin_all)
            s2_all = np.mean((vz_bin_all - v_mean_all) ** 2)
            ds2_all = (1.0 / (n_all * n_all * max(s2_all, 1e-12))) * np.sum(((vz_bin_all - v_mean_all) ** 2 - s2_all) ** 2)
            v_disp_all.append(np.sqrt(s2_all))
            d_v_disp_all.append(ds2_all ** 0.5)
        else:
            v_disp_all.append(np.nan)
            d_v_disp_all.append(np.nan)

    if return_diagnostics:
        diag = {
            "rj_kpc": rj,
            "m_enclosed": m_cluster,
            "m_bound": float(np.sum(m[is_bound])),
            "n_bound": int(np.sum(is_bound)),
            "n_potential_escapers": int(np.sum(is_potential_escaper)),
            "converged": converged,
            "iterations": n_iter,
            "center_kpc": (X_cm, Y_cm, Z_cm),
            "velocity_kms": (VX_cm, VY_cm, VZ_cm),
        }
        return is_bound, is_potential_escaper, diag

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
