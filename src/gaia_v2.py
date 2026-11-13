"""Gaia DR3 realistic forward-modeling and feature standardization for Study 2 V2.

This module provides an isolated V2 observational forward model without modifying
any Study 2 V1 modules. Key physical upgrades over V1:
1. Stellar luminosity function (Kroupa 2001 IMF + old halo RGB/MSTO branch)
   and realistic Gaia DR3 apparent magnitude limit (G <= 20.7).
2. Star-by-star heteroscedastic Lindegren et al. (2021) Gaia DR3 astrometric
   uncertainties computed directly from each star's apparent G magnitude.
3. Linear signed parallax space (varpi_obs = varpi_true + eps) preserving negative
   noisy parallaxes and completely eliminating biased 1/varpi_obs inversions.
4. Per-star astrometric uncertainty and parallax SNR side-channel features.
5. Local Milky Way halo/thick-disk contaminant star synthesizer in tangent sky
   and kinematic space for realistic cluster-versus-field discrimination.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


DEFAULT_G_MAX: float = 20.7
DEFAULT_MIN_VISIBLE_STARS: int = 64

# Physical reference scales for [dRA* (deg), dDec (deg), plx (mas), pmRA (mas/yr), pmDec (mas/yr), vlos (km/s)]
REF_SCALES_V2: Tuple[float, ...] = (2.0, 2.0, 0.1, 1.0, 1.0, 15.0)
OBS_NAMES_V2: Tuple[str, ...] = ("RA", "Dec", "Parallax", "pmRA", "pmDec", "Vlos")

# Feature-subset mappings matching Study 2 Step 6 ablation tags (5D astrometry = [0,1,2,3,4])
OBSERVABLE_SUBSETS: Dict[str, Tuple[int, ...]] = {
    "5obs": (0, 1, 2, 3, 4),
    "4obs_no_RA": (1, 2, 3, 4),
    "4obs_no_Dec": (0, 2, 3, 4),
    "4obs_no_Parallax": (0, 1, 3, 4),
    "4obs_no_pmRA": (0, 1, 2, 4),
    "4obs_no_pmDec": (0, 1, 2, 3),
    "6obs": (0, 1, 2, 3, 4, 5),
}


def _sample_powerlaw(
    n: int,
    alpha: float,
    x_min: float,
    x_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inverse-CDF sampler for p(x) proportional to x^{-alpha} on [x_min, x_max]."""
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    u = rng.uniform(0.0, 1.0, size=n)
    one_minus_a = 1.0 - alpha
    if abs(one_minus_a) < 1e-8:
        return x_min * (x_max / x_min) ** u
    a_min = x_min ** one_minus_a
    a_max = x_max ** one_minus_a
    return (a_min + u * (a_max - a_min)) ** (1.0 / one_minus_a)


def sample_cluster_abs_magnitudes_g(
    n_stars: int,
    rng: Optional[np.random.Generator] = None,
    giant_fraction: float = 0.12,
) -> np.ndarray:
    """Sample Gaia G-band absolute magnitudes M_G for an old metal-poor globular cluster.

    Combines:
    - Main-sequence stars (0.15-0.84 M_sun) drawn from a Kroupa (2001) broken
      power-law IMF (alpha1 = 1.3 for m < 0.5, alpha2 = 2.3 for m >= 0.5),
      mapped onto an old halo main-sequence M_G(m) relation (4.2 <= M_G <= 10.0).
    - Turnoff, subgiant, and RGB stars spanning M_G in [-0.6, 4.2] drawn from a
      luminosity function weighted toward the turnoff/lower-RGB.

    Parameters
    ----------
    n_stars : int
        Number of cluster stars to sample.
    rng : np.random.Generator, optional
        Random generator instance.
    giant_fraction : float
        Fraction of stars in the MSTO/subgiant/RGB branch (default: 0.12).

    Returns
    -------
    np.ndarray
        Absolute magnitudes M_G of shape (n_stars,).
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_stars <= 0:
        return np.empty(0, dtype=np.float64)

    is_giant = rng.uniform(0.0, 1.0, size=n_stars) < giant_fraction
    n_giants = int(np.sum(is_giant))
    n_ms = n_stars - n_giants

    m_g = np.empty(n_stars, dtype=np.float64)

    if n_ms > 0:
        # Continuous Kroupa (2001) IMF normalization across [0.15, 0.50] and [0.50, 0.84]
        p1 = (0.50 ** (-0.3) - 0.15 ** (-0.3)) / (-0.3)
        p2 = (0.50 ** 1.0) * (0.84 ** (-1.3) - 0.50 ** (-1.3)) / (-1.3)
        w1 = p1 / (p1 + p2)
        low_mask = rng.uniform(0.0, 1.0, size=n_ms) < w1
        n_low = int(np.sum(low_mask))
        masses = np.empty(n_ms, dtype=np.float64)
        masses[low_mask] = _sample_powerlaw(n_low, 1.3, 0.15, 0.50, rng)
        masses[~low_mask] = _sample_powerlaw(n_ms - n_low, 2.3, 0.50, 0.84, rng)
        ms_mag = 4.2 - 7.8 * np.log10(np.clip(masses / 0.84, 0.1, 1.0))
        ms_mag += rng.normal(0.0, 0.15, size=n_ms)
        m_g[~is_giant] = np.clip(ms_mag, 4.0, 10.8)

    if n_giants > 0:
        # Exponential luminosity function toward brighter RGB tip: p(M) ~ 10^{0.32 M} on [-0.6, 4.2]
        beta = 0.32 * np.log(10.0)
        lo, hi = -0.6, 4.2
        u = rng.uniform(0.0, 1.0, size=n_giants)
        e_lo, e_hi = np.exp(beta * lo), np.exp(beta * hi)
        giant_mag = np.log(e_lo + u * (e_hi - e_lo)) / beta
        m_g[is_giant] = np.clip(giant_mag, -0.8, 4.2)

    return m_g


def gaia_dr3_astrometric_sigmas(
    g_mag: np.ndarray,
    dec_deg: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute Lindegren et al. (2021) Gaia DR3 per-star 1-sigma astrometric errors.

    Returns dictionary with keys:
    - 'sigma_ra_deg': 1-sigma error in RA [deg], converted from sigma_alpha* / cos(dec)
    - 'sigma_dec_deg': 1-sigma error in Dec [deg]
    - 'sigma_plx_mas': 1-sigma error in parallax [mas]
    - 'sigma_pmra_mas_yr': 1-sigma error in pmRA*cos(dec) [mas/yr]
    - 'sigma_pmdec_mas_yr': 1-sigma error in pmDec [mas/yr]
    """
    g = np.asarray(g_mag, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)

    z_floor = 10.0 ** (0.4 * (13.0 - 15.0))
    z = np.maximum(10.0 ** (0.4 * (g - 15.0)), z_floor)
    sigma_0_uas = np.sqrt(40.0 + 800.0 * z + 30.0 * (z ** 2))
    sigma_plx_mas = sigma_0_uas / 1000.0

    cos_dec = np.maximum(np.abs(np.cos(np.deg2rad(dec))), 0.05)
    mas_to_deg = 1.0 / (3600.0 * 1000.0)

    return {
        "sigma_ra_deg": (0.80 * sigma_plx_mas * mas_to_deg) / cos_dec,
        "sigma_dec_deg": 0.70 * sigma_plx_mas * mas_to_deg,
        "sigma_plx_mas": sigma_plx_mas,
        "sigma_pmra_mas_yr": 1.03 * sigma_plx_mas,
        "sigma_pmdec_mas_yr": 0.89 * sigma_plx_mas,
    }


def encode_uncertainty_features_v2(
    plx_mas: np.ndarray,
    sigma_plx_mas: np.ndarray,
    sigma_pmra_mas_yr: np.ndarray,
    sigma_pmdec_mas_yr: np.ndarray,
) -> np.ndarray:
    """Encode per-star astrometric error magnitudes and signed parallax SNR into 4 bounded channels.

    Channels:
    0: log1p(sigma_plx / 0.1 mas)
    1: log1p(sigma_pmRA / 0.2 mas/yr)
    2: log1p(sigma_pmDec / 0.2 mas/yr)
    3: clip(plx / max(sigma_plx, 1e-4), -10, 50) / 10.0
    """
    sig_p = np.maximum(np.asarray(sigma_plx_mas, dtype=np.float64), 1e-4)
    sig_ra = np.maximum(np.asarray(sigma_pmra_mas_yr, dtype=np.float64), 1e-4)
    sig_dec = np.maximum(np.asarray(sigma_pmdec_mas_yr, dtype=np.float64), 1e-4)
    plx = np.asarray(plx_mas, dtype=np.float64)

    snr = np.clip(plx / sig_p, -10.0, 50.0) / 10.0
    feats = np.column_stack(
        (
            np.log1p(sig_p / 0.1),
            np.log1p(sig_ra / 0.2),
            np.log1p(sig_dec / 0.2),
            snr,
        )
    )
    return feats.astype(np.float32)


def mask_uncertainty_features(
    err_x: np.ndarray,
    selected_features: Sequence[int],
) -> np.ndarray:
    """Zero out uncertainty channels linked to omitted astrometric dimensions to prevent leakage.

    In 5D astrometry [0:RA, 1:Dec, 2:Parallax, 3:pmRA, 4:pmDec]:
    - Error channels 0 (sigma_plx) and 3 (plx_snr) require index 2 (Parallax).
    - Error channel 1 (sigma_pmra) requires index 3 (pmRA).
    - Error channel 2 (sigma_pmdec) requires index 4 (pmDec).
    """
    selected_set = set(int(i) for i in selected_features)
    out = err_x.copy()
    if 2 not in selected_set:
        out[..., 0] = 0.0
        out[..., 3] = 0.0
    if 3 not in selected_set:
        out[..., 1] = 0.0
    if 4 not in selected_set:
        out[..., 2] = 0.0
    return out


def convert_obs_v1_to_v2_linear_plx(obs_6d: np.ndarray) -> np.ndarray:
    """Return copy of [RA, Dec, D_kpc, pmRA, pmDec, vlos] with D_kpc replaced by linear parallax [mas]."""
    out = np.array(obs_6d, dtype=np.float64, copy=True)
    d_kpc = np.clip(out[:, 2], 1e-3, None)
    out[:, 2] = 1.0 / d_kpc
    return out


def apply_gaia_selection_and_noise_v2(
    obs_6d_clean: np.ndarray,
    abs_mag_g: Optional[np.ndarray] = None,
    g_max: float = DEFAULT_G_MAX,
    min_visible_stars: int = DEFAULT_MIN_VISIBLE_STARS,
    apply_mag_cut: bool = True,
    add_noise: bool = True,
    sigma_vlos_kms: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Apply Gaia DR3 IMF apparent-magnitude selection and linear-parallax error injection.

    Parameters
    ----------
    obs_6d_clean : np.ndarray, shape (N, 6)
        Clean observables [RA_deg, Dec_deg, D_kpc, pmRA_mas_yr, pmDec_mas_yr, vlos_kms].
    abs_mag_g : np.ndarray, optional
        Per-star absolute magnitudes M_G. Sampled from Kroupa/RGB luminosity function if None.
    g_max : float
        Gaia DR3 apparent magnitude detection threshold (default: 20.7).
    min_visible_stars : int
        Floor on surviving stars so k-NN graph topology stays well-defined.
    apply_mag_cut : bool
        If True, retain only stars with G <= g_max (subject to min_visible_stars floor).
    add_noise : bool
        If True, perturb observables by Lindegren et al. (2021) Gaussian errors.
    sigma_vlos_kms : float
        Optional radial velocity Gaussian error [km/s].
    rng : np.random.Generator, optional
        Random generator instance.

    Returns
    -------
    obs_plx : np.ndarray, shape (N_vis, 6)
        [RA_deg, Dec_deg, plx_mas, pmRA_mas_yr, pmDec_mas_yr, vlos_kms].
    err_feats : np.ndarray, shape (N_vis, 4)
        Bounded uncertainty and parallax-SNR features.
    vis_indices : np.ndarray, shape (N_vis,)
        Indices into input `obs_6d_clean` of surviving visible stars.
    g_app_vis : np.ndarray, shape (N_vis,)
        Apparent G magnitudes of surviving visible stars.
    meta : dict
        Summary diagnostics (`raw_vis_frac`, `cut_relaxed`, `median_g`, `median_sigma_plx`).
    """
    if rng is None:
        rng = np.random.default_rng()

    obs_raw = np.asarray(obs_6d_clean, dtype=np.float64)
    n_total = len(obs_raw)
    if abs_mag_g is None:
        abs_mag_g = sample_cluster_abs_magnitudes_g(n_total, rng=rng)
    else:
        abs_mag_g = np.asarray(abs_mag_g, dtype=np.float64)

    d_kpc = np.clip(obs_raw[:, 2], 1e-3, None)
    g_app = abs_mag_g + 5.0 * np.log10(d_kpc * 1000.0) - 5.0

    cut_relaxed = False
    if apply_mag_cut:
        vis_mask = g_app <= float(g_max)
        raw_vis_count = int(np.sum(vis_mask))
        floor = min(int(min_visible_stars), n_total)
        if raw_vis_count < floor:
            vis_indices = np.sort(np.argsort(g_app)[:floor])
            cut_relaxed = True
        else:
            vis_indices = np.flatnonzero(vis_mask)
    else:
        raw_vis_count = n_total
        vis_indices = np.arange(n_total, dtype=np.int64)

    obs_vis = convert_obs_v1_to_v2_linear_plx(obs_raw[vis_indices])
    g_app_vis = g_app[vis_indices]
    sigmas = gaia_dr3_astrometric_sigmas(g_app_vis, obs_vis[:, 1])

    if add_noise:
        obs_vis[:, 0] = (obs_vis[:, 0] + rng.normal(0.0, sigmas["sigma_ra_deg"])) % 360.0
        obs_vis[:, 1] = np.clip(obs_vis[:, 1] + rng.normal(0.0, sigmas["sigma_dec_deg"]), -89.999, 89.999)
        obs_vis[:, 2] = obs_vis[:, 2] + rng.normal(0.0, sigmas["sigma_plx_mas"])
        obs_vis[:, 3] = obs_vis[:, 3] + rng.normal(0.0, sigmas["sigma_pmra_mas_yr"])
        obs_vis[:, 4] = obs_vis[:, 4] + rng.normal(0.0, sigmas["sigma_pmdec_mas_yr"])
        if sigma_vlos_kms > 0.0:
            obs_vis[:, 5] = obs_vis[:, 5] + rng.normal(0.0, float(sigma_vlos_kms), size=len(vis_indices))

    err_feats = encode_uncertainty_features_v2(
        obs_vis[:, 2],
        sigmas["sigma_plx_mas"],
        sigmas["sigma_pmra_mas_yr"],
        sigmas["sigma_pmdec_mas_yr"],
    )

    meta = {
        "raw_vis_frac": float(raw_vis_count / max(n_total, 1)),
        "cut_relaxed": bool(cut_relaxed),
        "median_g": float(np.median(g_app_vis)),
        "median_sigma_plx": float(np.median(sigmas["sigma_plx_mas"])),
    }
    return obs_vis, err_feats, vis_indices, g_app_vis, meta


def _circular_sky_center(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[float, float]:
    """Compute outlier-robust circular S^1 median sky center (ra_c, dec_c) in degrees."""
    ra_rad = np.deg2rad(ra_deg)
    ra_c = float(
        np.rad2deg(np.arctan2(np.median(np.sin(ra_rad)), np.median(np.cos(ra_rad)))) % 360.0
    )
    dec_c = float(np.median(dec_deg))
    return ra_c, dec_c


def sample_mw_field_contaminants(
    stream_obs_plx: np.ndarray,
    contaminant_frac: float,
    g_max: float = DEFAULT_G_MAX,
    add_noise: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize foreground/background Milky Way halo/thick-disk stars inside the stream FOV.

    Parameters
    ----------
    stream_obs_plx : np.ndarray, shape (N_stream, 6)
        Cluster/stream observables [RA, Dec, plx_mas, pmRA, pmDec, vlos] (clean or noisy).
    contaminant_frac : float
        Target field fraction in the final combined catalog, in [0.0, 0.85].
        Number of added field stars is round(N_stream * f / (1 - f)).
    g_max : float
        Apparent G limit matching Gaia visibility.
    add_noise : bool
        If True, add Lindegren DR3 measurement errors to field stars; if False,
        return noiseless field stars paired with exact Gaia uncertainty vectors
        matching Curriculum Stage 1.
    rng : np.random.Generator, optional
        Random generator instance.

    Returns
    -------
    field_obs_plx : np.ndarray, shape (N_field, 6)
    field_err_feats : np.ndarray, shape (N_field, 4)
    field_g_app : np.ndarray, shape (N_field,)
    """
    if rng is None:
        rng = np.random.default_rng()

    f = float(np.clip(contaminant_frac, 0.0, 0.85))
    n_stream = len(stream_obs_plx)
    n_field = int(np.round(n_stream * f / max(1.0 - f, 1e-6))) if f > 0.0 else 0
    if n_field <= 0 or n_stream == 0:
        return (
            np.empty((0, 6), dtype=np.float64),
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float64),
        )

    ra_c, dec_c = _circular_sky_center(stream_obs_plx[:, 0], stream_obs_plx[:, 1])
    cos_dec_c = max(float(np.cos(np.deg2rad(dec_c))), 0.1)
    d_ra_star = (((stream_obs_plx[:, 0] - ra_c + 180.0) % 360.0) - 180.0) * cos_dec_c
    d_dec = stream_obs_plx[:, 1] - dec_c
    r_sky = np.hypot(d_ra_star, d_dec)
    r_fov = float(np.clip(np.percentile(r_sky, 95) * 1.25, 1.5, 25.0))

    # Sample field stars uniformly in 2D sky footprint
    theta_r = r_fov * np.sqrt(rng.uniform(0.0, 1.0, size=n_field))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_field)
    f_dra_star = theta_r * np.cos(phi)
    f_ddec = theta_r * np.sin(phi)

    f_dec = np.clip(dec_c + f_ddec, -89.999, 89.999)
    f_ra = (ra_c + f_dra_star / cos_dec_c) % 360.0

    # Halo + thick-disk distance/parallax distribution: log-uniform in [1.5, 45.0] kpc
    log_d = rng.uniform(np.log10(1.5), np.log10(45.0), size=n_field)
    f_plx = 1.0 / (10.0 ** log_d)

    # Halo/disk proper-motion background overlapping stream region
    med_pmra = float(np.median(stream_obs_plx[:, 3]))
    med_pmdec = float(np.median(stream_obs_plx[:, 4]))
    med_vlos = float(np.median(stream_obs_plx[:, 5]))
    std_pmra = max(float(np.std(stream_obs_plx[:, 3])) * 2.5, 2.5)
    std_pmdec = max(float(np.std(stream_obs_plx[:, 4])) * 2.5, 2.5)

    # Broad halo mixture (centered near stream reflex drift with broad scatter)
    drift_ra = rng.normal(0.0, 2.0)
    drift_dec = rng.normal(0.0, 2.0)
    f_pmra = rng.normal(med_pmra + drift_ra, std_pmra, size=n_field)
    f_pmdec = rng.normal(med_pmdec + drift_dec, std_pmdec, size=n_field)
    f_vlos = rng.normal(med_vlos, 110.0, size=n_field)

    # Galactic star counts steepen toward faint limit: p(G) ~ 10^{0.28 G} on [15.5, g_max]
    g_lo, g_hi = 15.5, float(g_max)
    beta = 0.28 * np.log(10.0)
    u_g = rng.uniform(0.0, 1.0, size=n_field)
    f_g_app = np.log(np.exp(beta * g_lo) + u_g * (np.exp(beta * g_hi) - np.exp(beta * g_lo))) / beta

    sigmas = gaia_dr3_astrometric_sigmas(f_g_app, f_dec)
    field_obs = np.column_stack((f_ra, f_dec, f_plx, f_pmra, f_pmdec, f_vlos))

    if add_noise:
        field_obs[:, 0] = (field_obs[:, 0] + rng.normal(0.0, sigmas["sigma_ra_deg"])) % 360.0
        field_obs[:, 1] = np.clip(field_obs[:, 1] + rng.normal(0.0, sigmas["sigma_dec_deg"]), -89.999, 89.999)
        field_obs[:, 2] = field_obs[:, 2] + rng.normal(0.0, sigmas["sigma_plx_mas"])
        field_obs[:, 3] = field_obs[:, 3] + rng.normal(0.0, sigmas["sigma_pmra_mas_yr"])
        field_obs[:, 4] = field_obs[:, 4] + rng.normal(0.0, sigmas["sigma_pmdec_mas_yr"])

    field_err = encode_uncertainty_features_v2(
        field_obs[:, 2],
        sigmas["sigma_plx_mas"],
        sigmas["sigma_pmra_mas_yr"],
        sigmas["sigma_pmdec_mas_yr"],
    )
    return field_obs, field_err, f_g_app


def standardise_matrix_v2(
    matrix_plx: np.ndarray,
    ref_scales: Tuple[float, ...] = REF_SCALES_V2,
    sky_center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Standardize [RA_deg, Dec_deg, plx_mas, pmRA, pmDec, vlos] without parallax inversion.

    - RA/Dec are projected to tangent-plane offsets (dRA*cos(Dec_c), dDec) around
      the circular S^1 median sky coordinate (or explicit `sky_center`) and scaled
      by `ref_scales[0:2]`.
    - Parallax is kept strictly linear and uncentered (`plx_mas / ref_scales[2]`),
      preserving negative noisy measurements and avoiding contaminant-induced shifts.
    - Proper motions and v_los are median-centered and scaled by `ref_scales[3:6]`.
    """
    mat = np.asarray(matrix_plx, dtype=np.float64)
    n_cols = mat.shape[1]
    out = np.empty_like(mat, dtype=np.float32)

    if sky_center is None:
        ra_c, dec_c = _circular_sky_center(mat[:, 0], mat[:, 1])
    else:
        ra_c, dec_c = float(sky_center[0]), float(sky_center[1])

    cos_dec_c = max(float(np.cos(np.deg2rad(dec_c))), 0.05)
    d_ra = ((mat[:, 0] - ra_c + 180.0) % 360.0) - 180.0
    out[:, 0] = (d_ra * cos_dec_c / ref_scales[0]).astype(np.float32)
    out[:, 1] = ((mat[:, 1] - dec_c) / ref_scales[1]).astype(np.float32)
    out[:, 2] = (mat[:, 2] / ref_scales[2]).astype(np.float32)

    medians = np.zeros(n_cols, dtype=np.float64)
    medians[0] = ra_c
    medians[1] = dec_c
    medians[2] = 0.0

    for col in range(3, n_cols):
        med = float(np.median(mat[:, col]))
        medians[col] = med
        scale = ref_scales[col] if col < len(ref_scales) else float(np.std(mat[:, col]) + 1e-6)
        out[:, col] = ((mat[:, col] - med) / scale).astype(np.float32)

    meta = {
        "medians": medians,
        "scales": np.asarray(ref_scales[:n_cols], dtype=np.float64),
    }
    return out, meta
