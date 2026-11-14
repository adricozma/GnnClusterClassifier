"""Extract Study 2 V2 3-class labels, Gaia DR3 observables, uncertainties, and MW contaminants.

Reads raw SWIFT HDF5 snapshots (read-only from `/scratch/cozma/study2/outputs_unbiased`),
computes dynamical Jacobi-radius labels (0: Unbound/Field, 1: Potential Escaper, 2: Bound Core)
via `src/bound_labels.py`, and generates paired Clean and Gaia DR3 Noisy realizations under
SO(3) rotations with:
- Kroupa (2001) + RGB/MSTO stellar luminosity sampling and Gaia limit G <= 20.7
- Linear signed parallax space (no 1/varpi inversion)
- Star-by-star Lindegren et al. (2021) astrometric errors and SNR channels
- Milky Way halo/thick-disk field contaminant mixing across [0%, 55%] per realization
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bound_labels import compute_bound_stars
from gaia_v2 import (
    DEFAULT_G_MAX,
    DEFAULT_MIN_VISIBLE_STARS,
    REF_SCALES_V2,
    apply_gaia_selection_and_noise_v2,
    encode_uncertainty_features_v2,
    gaia_dr3_astrometric_sigmas,
    sample_cluster_abs_magnitudes_g,
    sample_mw_field_contaminants,
    standardise_matrix_v2,
)
from observables_matrix import get_observables_matrix, rotate_data
from plot_swift_output import load_snapshot


def _is_valid_npz(path: str, min_bytes: int = 2048) -> bool:
    return bool(os.path.isfile(path) and os.path.getsize(path) >= min_bytes)


def _add_gaia_noise_to_catalog(
    obs_plx: np.ndarray,
    g_app: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perturb an exact Gaia [RA, Dec, plx, pmRA, pmDec, vlos] table with Lindegren DR3 errors."""
    out = np.array(obs_plx, dtype=np.float64, copy=True)
    if len(out) == 0:
        return out, np.empty((0, 4), dtype=np.float32)

    sigmas = gaia_dr3_astrometric_sigmas(g_app, out[:, 1])
    out[:, 0] = (out[:, 0] + rng.normal(0.0, sigmas["sigma_ra_deg"])) % 360.0
    out[:, 1] = np.clip(out[:, 1] + rng.normal(0.0, sigmas["sigma_dec_deg"]), -89.999, 89.999)
    out[:, 2] = out[:, 2] + rng.normal(0.0, sigmas["sigma_plx_mas"])
    out[:, 3] = out[:, 3] + rng.normal(0.0, sigmas["sigma_pmra_mas_yr"])
    out[:, 4] = out[:, 4] + rng.normal(0.0, sigmas["sigma_pmdec_mas_yr"])

    err = encode_uncertainty_features_v2(
        out[:, 2],
        sigmas["sigma_plx_mas"],
        sigmas["sigma_pmra_mas_yr"],
        sigmas["sigma_pmdec_mas_yr"],
    )
    return out, err


def _process_snapshot(task: Tuple) -> Tuple[str, bool]:
    (
        fname,
        snap_stem,
        out_path,
        n_rotations,
        g_max,
        min_vis_stars,
        contaminant_min,
        contaminant_max,
        pure_prob,
        box_shift,
        seed,
        overwrite,
    ) = task

    if not overwrite and _is_valid_npz(out_path):
        return snap_stem, True

    rng = np.random.default_rng(seed)
    pos_raw, vel_raw, mass_raw, _ = load_snapshot(fname, shift=box_shift)
    masses = (
        np.ones(len(pos_raw), dtype=float)
        if mass_raw is None
        else np.asarray(mass_raw, dtype=float)
    )

    is_bound, is_pe, _ = compute_bound_stars(
        pos_raw[:, 0],
        pos_raw[:, 1],
        pos_raw[:, 2],
        vel_raw[:, 0],
        vel_raw[:, 1],
        vel_raw[:, 2],
        masses,
        return_diagnostics=True,
    )

    # 3-class taxonomy: 0 = Unbound/Field, 1 = Potential Escaper, 2 = Bound Core
    y_cluster_all = np.zeros(len(pos_raw), dtype=np.int8)
    y_cluster_all[is_pe] = 1
    y_cluster_all[is_bound] = 2

    # Assign intrinsic stellar luminosity M_G once per particle across rotations
    abs_mag_g_all = sample_cluster_abs_magnitudes_g(len(pos_raw), rng=rng)

    payload: Dict[str, np.ndarray] = {
        "n_rotations": np.array(n_rotations, dtype=np.int32),
        "g_max": np.array(g_max, dtype=np.float32),
    }

    for r in range(n_rotations):
        pos_rot, vel_rot = rotate_data(pos_raw, vel_raw, rng=rng)
        matrix_obs = get_observables_matrix(pos_rot, vel_rot)

        clean_stream_plx, clean_stream_err, vis_idx, g_stream_vis, meta = (
            apply_gaia_selection_and_noise_v2(
                matrix_obs,
                abs_mag_g=abs_mag_g_all,
                g_max=g_max,
                min_visible_stars=min_vis_stars,
                apply_mag_cut=True,
                add_noise=False,
                rng=rng,
            )
        )
        noisy_stream_plx, noisy_stream_err = _add_gaia_noise_to_catalog(
            clean_stream_plx, g_stream_vis, rng=rng
        )
        y_stream_vis = y_cluster_all[vis_idx]

        # Sample contaminant fraction per realization (keep some pure-cluster realizations)
        if contaminant_max > 0.0 and rng.uniform(0.0, 1.0) >= pure_prob:
            f_contam = float(rng.uniform(contaminant_min, contaminant_max))
        else:
            f_contam = 0.0

        field_clean_plx, field_clean_err, f_g_app = sample_mw_field_contaminants(
            clean_stream_plx,
            contaminant_frac=f_contam,
            g_max=g_max,
            add_noise=False,
            rng=rng,
        )
        field_noisy_plx, field_noisy_err = _add_gaia_noise_to_catalog(
            field_clean_plx, f_g_app, rng=rng
        )

        n_field = len(field_clean_plx)
        if n_field > 0:
            clean_plx = np.vstack((clean_stream_plx, field_clean_plx))
            clean_err = np.vstack((clean_stream_err, field_clean_err))
            noisy_plx = np.vstack((noisy_stream_plx, field_noisy_plx))
            noisy_err = np.vstack((noisy_stream_err, field_noisy_err))
            y_comb = np.concatenate((y_stream_vis, np.zeros(n_field, dtype=np.int8)))
            is_field = np.concatenate(
                (np.zeros(len(y_stream_vis), dtype=bool), np.ones(n_field, dtype=bool))
            )
        else:
            clean_plx = clean_stream_plx
            clean_err = clean_stream_err
            noisy_plx = noisy_stream_plx
            noisy_err = noisy_stream_err
            y_comb = y_stream_vis
            is_field = np.zeros(len(y_stream_vis), dtype=bool)

        perm = rng.permutation(len(y_comb))
        clean_plx = clean_plx[perm]
        clean_err = clean_err[perm]
        noisy_plx = noisy_plx[perm]
        noisy_err = noisy_err[perm]
        y_comb = y_comb[perm]
        is_field = is_field[perm]

        stream_mask = ~is_field
        clean_x_std, _ = standardise_matrix_v2(
            clean_plx, ref_scales=REF_SCALES_V2, reference_mask=stream_mask
        )
        noisy_x_std, _ = standardise_matrix_v2(
            noisy_plx, ref_scales=REF_SCALES_V2, reference_mask=stream_mask
        )

        payload[f"clean_x_{r}"] = clean_x_std.astype(np.float32)
        payload[f"clean_err_{r}"] = clean_err.astype(np.float32)
        payload[f"noisy_x_{r}"] = noisy_x_std.astype(np.float32)
        payload[f"noisy_err_{r}"] = noisy_err.astype(np.float32)
        payload[f"y_{r}"] = y_comb.astype(np.int8)
        payload[f"is_field_{r}"] = is_field
        payload[f"vis_frac_{r}"] = np.array(meta["raw_vis_frac"], dtype=np.float32)
        payload[f"contam_frac_{r}"] = np.array(f_contam, dtype=np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + f".tmp.{os.getpid()}.npz"
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, out_path)
    return snap_stem, False


def _discover_ic_folders(sim_dir: str, ic_index: int) -> List[str]:
    ics_dir = os.path.join(ROOT_DIR, "make_ICs", "initial_conditions_unbiased")
    if ic_index >= 0 and os.path.isdir(ics_dir):
        canonical_ic_files = sorted(
            [f for f in os.listdir(ics_dir) if f.startswith("IC_") and f.endswith(".hdf5")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if ic_index >= len(canonical_ic_files):
            return []
        return [os.path.splitext(canonical_ic_files[ic_index])[0]]

    if not os.path.isdir(sim_dir):
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    ic_folders = sorted(
        [
            f
            for f in os.listdir(sim_dir)
            if f.startswith("IC_") and os.path.isdir(os.path.join(sim_dir, f))
        ],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
    )
    if ic_index >= 0:
        return [ic_folders[ic_index]] if ic_index < len(ic_folders) else []
    return ic_folders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Study 2 V2 Gaia DR3 observables, uncertainties, and 3-class labels."
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        default="/scratch/cozma/study2/outputs_unbiased",
        help="Read-only SWIFT HDF5 outputs directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/scratch/cozma/study2_v2/obs_unbiased",
        help="Output V2 observables directory.",
    )
    parser.add_argument("--n_rotations", type=int, default=5)
    parser.add_argument("--g_max", type=float, default=DEFAULT_G_MAX)
    parser.add_argument("--min_vis_stars", type=int, default=DEFAULT_MIN_VISIBLE_STARS)
    parser.add_argument("--contaminant_min", type=float, default=0.05)
    parser.add_argument("--contaminant_max", type=float, default=0.55)
    parser.add_argument(
        "--pure_prob",
        type=float,
        default=0.15,
        help="Probability of zero MW field contamination for a given rotation.",
    )
    parser.add_argument("--box_shift", type=float, default=500.0)
    parser.add_argument("--ic_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_NTASKS", 16))),
    )
    args = parser.parse_args()

    ic_folders = _discover_ic_folders(args.sim_dir, args.ic_index)
    if not ic_folders:
        print(f"IC index {args.ic_index} out of range. Exiting cleanly.")
        return

    tasks = []
    for ic_name in ic_folders:
        ic_sim_path = os.path.join(args.sim_dir, ic_name)
        if not os.path.isdir(ic_sim_path):
            continue
        snaps = sorted(
            f
            for f in os.listdir(ic_sim_path)
            if f.startswith("snapshot_") and f.endswith(".hdf5")
        )
        ic_out_path = os.path.join(args.out_dir, ic_name)
        for snap_idx, snap_file in enumerate(snaps):
            snap_stem = os.path.splitext(snap_file)[0]
            out_npz = os.path.join(ic_out_path, f"v2_obs_{snap_stem}.npz")
            seed_bytes = f"{ic_name}:{snap_stem}:{args.seed}".encode("utf-8")
            ic_seed = int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=4).digest(), "big")
            tasks.append(
                (
                    os.path.join(ic_sim_path, snap_file),
                    snap_stem,
                    out_npz,
                    args.n_rotations,
                    args.g_max,
                    args.min_vis_stars,
                    args.contaminant_min,
                    args.contaminant_max,
                    args.pure_prob,
                    args.box_shift,
                    ic_seed + snap_idx,
                    args.overwrite,
                )
            )

    print(
        f"V2 extraction: {len(tasks)} snapshot(s) across {len(ic_folders)} IC folder(s) "
        f"(G<={args.g_max}, contam=[{args.contaminant_min},{args.contaminant_max}], workers={args.num_workers})."
    )
    if not tasks:
        return

    processed = 0
    skipped = 0
    if args.num_workers <= 1:
        for t in tasks:
            _, was_skipped = _process_snapshot(t)
            skipped += int(was_skipped)
            processed += int(not was_skipped)
    else:
        with ProcessPoolExecutor(max_workers=min(args.num_workers, len(tasks))) as pool:
            futs = [pool.submit(_process_snapshot, t) for t in tasks]
            for fut in as_completed(futs):
                _, was_skipped = fut.result()
                skipped += int(was_skipped)
                processed += int(not was_skipped)

    print(f"V2 extraction finished: {processed} generated, {skipped} skipped (cached).")


if __name__ == "__main__":
    main()
