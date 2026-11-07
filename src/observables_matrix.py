import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from scipy.spatial.transform import Rotation as R


def get_observables_matrix(pos, vel):
    """
    Projects 3D Galactocentric Cartesian positions and velocities into 6D heliocentric ICRS observables.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 3)
        Galactocentric Cartesian coordinates (x, y, z) in kpc.
    vel : np.ndarray, shape (N, 3)
        Galactocentric Cartesian velocities (vx, vy, vz) in km/s.

    Returns
    -------
    obs_matrix : np.ndarray, shape (N, 6)
        Columns: [RA (deg), Dec (deg), heliocentric distance D (kpc),
                  pm_ra_cosdec (mas/yr), pm_dec (mas/yr), v_los (km/s)].
    """
    sim_coords = SkyCoord(
        x=pos[:, 0] * u.kpc,
        y=pos[:, 1] * u.kpc,
        z=pos[:, 2] * u.kpc,
        v_x=vel[:, 0] * (u.km / u.s),
        v_y=vel[:, 1] * (u.km / u.s),
        v_z=vel[:, 2] * (u.km / u.s),
        frame=Galactocentric(),
    )

    obs_coords = sim_coords.transform_to(ICRS())

    return np.column_stack((
        obs_coords.ra.degree,
        obs_coords.dec.degree,
        obs_coords.distance.kpc,
        obs_coords.pm_ra_cosdec.value,
        obs_coords.pm_dec.value,
        obs_coords.radial_velocity.value,
    ))


def rotate_data(pos, vel, rng=None):
    """
    Applies a uniform random 3D rotation sampled from the Haar measure on SO(3) (Shoemake 1992).
    """
    rot_matrix = R.random(random_state=rng).as_matrix()
    return np.dot(pos, rot_matrix.T), np.dot(vel, rot_matrix.T)


def inject_gaia_dr3_noise(matrix_obs, abs_mag_g=4.5, rng=None, min_dist_kpc=0.1, max_dist_kpc=200.0):
    """
    Injects realistic Gaia DR3 astrometric measurement uncertainties into the observables matrix
    following the official Gaia DR3 error model (Lindegren et al. 2021, A&A 649, A2, Table A.1).

    Parameters
    ----------
    matrix_obs : np.ndarray, shape (N, 6)
        Clean observables [RA (deg), Dec (deg), D (kpc), pm_ra_cosdec (mas/yr), pm_dec (mas/yr), v_los (km/s)].
    abs_mag_g : float
        Characteristic absolute G-band magnitude of the tracer stars (default: 4.5 for main-sequence turnoff stars).
    rng : np.random.Generator or int, optional
        Random number generator or seed for reproducibility.

    Returns
    -------
    noisy_matrix : np.ndarray, shape (N, 6)
        Observables perturbed by Gaussian astrometric errors.
    """
    rng = np.random.default_rng(rng)
    noisy = np.array(matrix_obs, dtype=float, copy=True)

    dist_kpc = np.clip(noisy[:, 2], min_dist_kpc, None)
    # Apparent G magnitude: G = M_G + 5 * log10(D_pc / 10) = M_G + 5 * log10(D_kpc) + 10
    g_mag = abs_mag_g + 5.0 * np.log10(dist_kpc) + 10.0

    # Gaia DR3 parallax uncertainty sigma_parallax in mas (Lindegren et al. 2021)
    z = np.maximum(10.0 ** (0.4 * (13.0 - 15.0)), 10.0 ** (0.4 * (g_mag - 15.0)))
    sigma_plx_mas = 1e-3 * np.sqrt(40.0 + 800.0 * z + 30.0 * (z ** 2))

    # Position uncertainties (converted from mas to deg) and proper motion uncertainties (mas/yr)
    mas_to_deg = 1.0 / 3.6e6
    sigma_ra_deg = 0.80 * sigma_plx_mas * mas_to_deg / np.maximum(np.cos(np.deg2rad(noisy[:, 1])), 1e-3)
    sigma_dec_deg = 0.70 * sigma_plx_mas * mas_to_deg
    sigma_pmra = 1.03 * sigma_plx_mas
    sigma_pmdec = 0.89 * sigma_plx_mas

    # Perturb sky positions and proper motions
    noisy[:, 0] = (noisy[:, 0] + rng.normal(0.0, sigma_ra_deg)) % 360.0
    noisy[:, 1] = np.clip(noisy[:, 1] + rng.normal(0.0, sigma_dec_deg), -90.0, 90.0)
    noisy[:, 3] += rng.normal(0.0, sigma_pmra)
    noisy[:, 4] += rng.normal(0.0, sigma_pmdec)

    # Perturb parallax plx = 1 / D_kpc (mas) and convert back to distance (clipped to physical bounds)
    plx_true_mas = 1.0 / dist_kpc
    plx_obs_mas = plx_true_mas + rng.normal(0.0, sigma_plx_mas)
    plx_min_mas = 1.0 / max_dist_kpc
    noisy[:, 2] = 1.0 / np.clip(plx_obs_mas, plx_min_mas, 1.0 / min_dist_kpc)

    return noisy


def standardise_matrix(matrix, mode="zscore", ref_scales=(2.0, 2.0, 1.5, 1.0, 1.0, 15.0)):
    """
    Standardizes the observables matrix [RA (deg), Dec (deg), D (kpc), pm_ra_cosdec (mas/yr), pm_dec (mas/yr), v_los (km/s)].

    In both 'zscore' and 'physical' modes, Right Ascension (RA) is first centered using the circular median
    on S^1 and wrapped into [-180, +180] deg with a local tangent-plane cos(Dec_c) projection. This prevents
    streams crossing the 0/360 deg meridian from being split into two artificial components.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, 6) or (N, 5)
        Raw observables matrix.
    mode : {'zscore', 'physical'}
        - 'zscore' (default, compatible with existing GNN methodology): Wrap-safe centering followed by
          per-snapshot standard deviation scaling.
        - 'physical': Wrap-safe median centering followed by division by fixed physical scales `ref_scales`.
    ref_scales : tuple of float
        Characteristic physical scales (deg, deg, kpc, mas/yr, mas/yr, km/s) used when mode='physical'.
    """
    matrix = np.asarray(matrix, dtype=float)
    work = matrix.copy()

    # Circular median for RA on S^1 to avoid 0/360 deg wrap-around discontinuities
    ra_rad = np.deg2rad(work[:, 0])
    ra_center_rad = np.arctan2(np.median(np.sin(ra_rad)), np.median(np.cos(ra_rad)))
    ra_center_deg = float(np.rad2deg(ra_center_rad) % 360.0)
    dec_center_deg = float(np.median(work[:, 1]))

    # Tangent-plane angular offset dRA * cos(Dec_c) in [-180, 180] deg
    cos_dec = max(float(np.cos(np.deg2rad(dec_center_deg))), 1e-4)
    work[:, 0] = ((work[:, 0] - ra_center_deg + 180.0) % 360.0 - 180.0) * cos_dec

    if mode == "zscore":
        medians = np.median(work, axis=0)
        mad_scale = 1.4826 * np.median(np.abs(work - medians), axis=0)
        raw_stds = np.std(work, axis=0)
        robust_scale = np.where(mad_scale > 1e-6, mad_scale, np.where(raw_stds > 1e-8, raw_stds, 1.0))

        # Winsorize extreme >6*MAD astrometric noise spikes (e.g. non-linear 1/parallax inversions)
        # so heavy-tailed measurement noise cannot inflate the variance and collapse the core-tail contrast.
        clipped = np.clip(work, medians - 6.0 * robust_scale, medians + 6.0 * robust_scale)
        means = np.mean(clipped, axis=0)
        stds = np.std(clipped, axis=0)
        stds = np.where(stds > 1e-8, stds, robust_scale)
        standardized = (clipped - means) / stds
        means_out = means.copy()
        means_out[0] = ra_center_deg
        return standardized, means_out, stds

    if mode == "physical":
        centers = np.median(work, axis=0)
        centered = work - centers
        scales = np.asarray(ref_scales[: work.shape[1]], dtype=float)
        centers_out = centers.copy()
        centers_out[0] = ra_center_deg
        return centered / scales, centers_out, scales

    raise ValueError(f"Unknown standardisation mode '{mode}'. Expected 'zscore' or 'physical'.")


def plot_observables(matrix_obs, output_name="rotation_check.png"):
    ra = matrix_obs[:, 0]
    dec = matrix_obs[:, 1]
    v_los = matrix_obs[:, 5]

    sc = plt.scatter(ra, dec, c=v_los, s=1, cmap="viridis", alpha=0.6)
    plt.colorbar(sc, label="v_{los} [km/s]")

    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("Observed projection (random rotation)")
    plt.grid(True, alpha=0.2)
    plt.gca().invert_xaxis()

    plt.savefig(output_name, bbox_inches="tight")
    plt.close()
