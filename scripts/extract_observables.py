"""Extract dynamical boundness labels and 5D/6D astrometric observable matrices from SWIFT HDF5 snapshots.

Supports multi-core CPU parallelization across snapshots (`--num_workers`), SLURM job-array
indexing (`--ic_index`), and simultaneous extraction of both noiseless (`--obs_dir`) and
Gaia DR3 noisy (`--obs_noisy_dir`) observable matrices from the same random SO(3) rotations.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bound_labels import compute_bound_stars
from observables_matrix import (
    get_observables_matrix,
    inject_gaia_dr3_noise,
    rotate_data,
    standardise_matrix,
)
from plot_swift_output import load_snapshot


def _process_single_snapshot(task):
    """Worker function to process a single HDF5 snapshot."""
    (
        fname,
        snap_stem,
        out_label_dir,
        out_obs_dir,
        out_obs_noisy_dir,
        n_rotations,
        norm_mode,
        add_gaia_noise,
        abs_mag_g,
        box_shift,
        seed,
    ) = task

    label_path = os.path.join(out_label_dir, f"B_L_{snap_stem}.npy")
    obs_path = os.path.join(out_obs_dir, f"obs_matrices_{snap_stem}.npy")
    obs_noisy_path = (
        os.path.join(out_obs_noisy_dir, f"obs_matrices_{snap_stem}.npy")
        if out_obs_noisy_dir
        else None
    )

    def _is_valid_file(p, min_bytes=512):
        return bool(p and os.path.isfile(p) and os.path.getsize(p) > min_bytes)

    skip = _is_valid_file(label_path, 512) and _is_valid_file(obs_path, 10000)
    if obs_noisy_path is not None:
        skip = skip and _is_valid_file(obs_noisy_path, 10000)

    if skip:
        return snap_stem, True

    rng = np.random.default_rng(seed)
    pos_raw, vel_raw, mass_raw, _ = load_snapshot(fname, shift=box_shift)
    masses = (
        np.ones(len(pos_raw), dtype=float)
        if mass_raw is None
        else np.asarray(mass_raw, dtype=float)
    )

    # 1. Compute exact bound star labels (Küpper et al. 2010a,b)
    is_bound, _, _ = compute_bound_stars(
        pos_raw[:, 0],
        pos_raw[:, 1],
        pos_raw[:, 2],
        vel_raw[:, 0],
        vel_raw[:, 1],
        vel_raw[:, 2],
        masses,
        return_diagnostics=True,
    )
    np.save(label_path, is_bound)

    # 2. Compute rotated 6D/5D observables matrices (clean and optionally Gaia DR3 noisy)
    clean_matrices = []
    noisy_matrices = []
    for _ in range(n_rotations):
        pos_rot, vel_rot = rotate_data(pos_raw, vel_raw)
        matrix_obs = get_observables_matrix(pos_rot, vel_rot)

        if out_obs_noisy_dir:
            x_clean, _, _ = standardise_matrix(matrix_obs, mode=norm_mode)
            clean_matrices.append(x_clean)

            matrix_noisy = inject_gaia_dr3_noise(
                matrix_obs, abs_mag_g=abs_mag_g, rng=rng
            )
            x_noisy, _, _ = standardise_matrix(matrix_noisy, mode=norm_mode)
            noisy_matrices.append(x_noisy)
        else:
            if add_gaia_noise:
                matrix_obs = inject_gaia_dr3_noise(
                    matrix_obs, abs_mag_g=abs_mag_g, rng=rng
                )
            x_ready, _, _ = standardise_matrix(matrix_obs, mode=norm_mode)
            clean_matrices.append(x_ready)

    np.save(obs_path, np.asarray(clean_matrices, dtype=np.float32))
    if obs_noisy_path is not None:
        np.save(obs_noisy_path, np.asarray(noisy_matrices, dtype=np.float32))

    return snap_stem, False


def main():
    parser = argparse.ArgumentParser(
        description="Extract dynamical boundness labels and 5D/6D astrometric observable matrices from SWIFT HDF5 snapshots."
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        default="/scratch/cozma/study2/outputs_unbiased",
        help="Directory containing IC_* simulation folders with snapshot_*.hdf5 files.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="/scratch/cozma/study2/labels_clean",
        help="Output directory for bound star boolean labels (B_L_snapshot_*.npy).",
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        default="/scratch/cozma/study2/obs_clean",
        help="Output directory for standardized observable matrices (obs_matrices_snapshot_*.npy).",
    )
    parser.add_argument(
        "--obs_noisy_dir",
        type=str,
        default="",
        help="Optional second output directory to simultaneously write Gaia DR3 noisy observables from the same rotations.",
    )
    parser.add_argument(
        "--n_rotations",
        type=int,
        default=5,
        help="Number of random 3D SO(3) rotations per snapshot for data augmentation.",
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="zscore",
        choices=["zscore", "physical"],
        help="Standardization mode ('zscore' or 'physical').",
    )
    parser.add_argument(
        "--add_gaia_noise",
        action="store_true",
        help="If set (and --obs_noisy_dir is empty), inject Gaia DR3 uncertainties into --obs_dir.",
    )
    parser.add_argument(
        "--abs_mag_g",
        type=float,
        default=4.5,
        help="Characteristic absolute G-band magnitude of tracer stars for the Gaia DR3 noise model.",
    )
    parser.add_argument(
        "--box_shift",
        type=float,
        default=500.0,
        help="Simulation box center offset in kpc to subtract from raw SWIFT coordinates.",
    )
    parser.add_argument(
        "--ic_index",
        type=int,
        default=-1,
        help="Optional 0-based index into sorted IC_* folders (for SLURM job arrays). Default -1 processes all.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK", os.environ.get("SLURM_NTASKS", 16)
            )
        ),
        help="Number of parallel CPU worker processes.",
    )
    args = parser.parse_args()

    ics_dir = os.path.join(ROOT_DIR, "make_ICs", "initial_conditions_unbiased")
    canonical_ic_files = sorted(
        [
            f
            for f in os.listdir(ics_dir)
            if f.startswith("IC_") and f.endswith(".hdf5")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    if args.ic_index >= 0:
        if args.ic_index >= len(canonical_ic_files):
            print(
                f"IC index {args.ic_index} out of range (total {len(canonical_ic_files)}). Exiting cleanly."
            )
            return
        target_ic_name = os.path.splitext(canonical_ic_files[args.ic_index])[0]
        ic_folders = [target_ic_name]
    else:
        if not os.path.exists(args.sim_dir):
            raise FileNotFoundError(f"Simulation directory not found: {args.sim_dir}")
        ic_folders = sorted(
            [
                f
                for f in os.listdir(args.sim_dir)
                if f.startswith("IC_") and os.path.isdir(os.path.join(args.sim_dir, f))
            ],
            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
        )

    print(
        f"Processing {len(ic_folders)} IC folder(s) from '{args.sim_dir}' using {args.num_workers} CPU workers."
    )

    tasks = []
    for ic_name in ic_folders:
        ic_path = os.path.join(args.sim_dir, ic_name)
        if not os.path.isdir(ic_path):
            raise FileNotFoundError(f"Simulation folder not found: {ic_path}")
        snapshots = sorted(
            [
                f
                for f in os.listdir(ic_path)
                if f.startswith("snapshot_") and f.endswith(".hdf5")
            ]
        )
        out_label_dir = os.path.join(args.labels_dir, ic_name)
        out_obs_dir = os.path.join(args.obs_dir, ic_name)
        os.makedirs(out_label_dir, exist_ok=True)
        os.makedirs(out_obs_dir, exist_ok=True)

        out_obs_noisy_dir = None
        if args.obs_noisy_dir:
            out_obs_noisy_dir = os.path.join(args.obs_noisy_dir, ic_name)
            os.makedirs(out_obs_noisy_dir, exist_ok=True)

        ic_num = int(ic_name.split("_")[1]) if ic_name.split("_")[1].isdigit() else 0
        for snap_idx, snap_file in enumerate(snapshots):
            snap_stem = os.path.splitext(snap_file)[0]
            fname = os.path.join(ic_path, snap_file)
            seed = 100000 + ic_num * 1000 + snap_idx
            tasks.append(
                (
                    fname,
                    snap_stem,
                    out_label_dir,
                    out_obs_dir,
                    out_obs_noisy_dir,
                    args.n_rotations,
                    args.norm_mode,
                    args.add_gaia_noise,
                    args.abs_mag_g,
                    args.box_shift,
                    seed,
                )
            )

    if args.num_workers <= 1:
        for idx, t in enumerate(tasks, 1):
            snap_stem, skipped = _process_single_snapshot(t)
            if idx % 25 == 0 or idx == len(tasks):
                print(f"[{idx}/{len(tasks)}] Completed {snap_stem} (skipped={skipped})")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(_process_single_snapshot, t) for t in tasks]
            for fut in as_completed(futures):
                snap_stem, skipped = fut.result()
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    print(
                        f"[{completed}/{len(tasks)}] Latest: {snap_stem} (skipped={skipped})"
                    )


if __name__ == "__main__":
    main()
