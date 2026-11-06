import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract dynamical boundness labels and 5D/6D astrometric observable matrices from SWIFT HDF5 snapshots."
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "outputs_SWIFT_jed", "withoutDF"),
        help="Directory containing IC_* simulation folders with snapshot_*.hdf5 files.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "labels_bound_stars"),
        help="Output directory for bound star boolean labels (B_L_snapshot_*.npy).",
    )
    parser.add_argument(
        "--obs_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "obs_all_stars"),
        help="Output directory for standardized observable matrices (obs_matrices_snapshot_*.npy).",
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
        help="If set, inject realistic Gaia DR3 astrometric uncertainties before standardization (for Stage 2 fine-tuning).",
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
    args = parser.parse_args()

    if not os.path.exists(args.sim_dir):
        raise FileNotFoundError(f"Simulation directory not found: {args.sim_dir}")

    ic_folders = sorted([f for f in os.listdir(args.sim_dir) if f.startswith("IC_")])
    print(f"Found {len(ic_folders)} IC directories in '{args.sim_dir}'.")

    for ic_name in ic_folders:
        ic_path = os.path.join(args.sim_dir, ic_name)
        if not os.path.isdir(ic_path):
            continue

        snapshots = sorted([f for f in os.listdir(ic_path) if f.startswith("snapshot_") and f.endswith(".hdf5")])
        out_label_dir = os.path.join(args.labels_dir, ic_name)
        out_obs_dir = os.path.join(args.obs_dir, ic_name)
        os.makedirs(out_label_dir, exist_ok=True)
        os.makedirs(out_obs_dir, exist_ok=True)

        print(f"Processing {ic_name} ({len(snapshots)} snapshots)...")

        for snap_file in snapshots:
            snap_stem = os.path.splitext(snap_file)[0]
            fname = os.path.join(ic_path, snap_file)

            pos_raw, vel_raw, mass_raw, _ = load_snapshot(fname, shift=args.box_shift)
            masses = np.ones(len(pos_raw), dtype=float) if mass_raw is None else np.asarray(mass_raw, dtype=float)

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
            np.save(os.path.join(out_label_dir, f"B_L_{snap_stem}.npy"), is_bound)

            # 2. Compute rotated 6D/5D observables matrices
            final_matrices = []
            for _ in range(args.n_rotations):
                pos_rot, vel_rot = rotate_data(pos_raw, vel_raw)
                matrix_obs = get_observables_matrix(pos_rot, vel_rot)
                if args.add_gaia_noise:
                    matrix_obs = inject_gaia_dr3_noise(matrix_obs, abs_mag_g=args.abs_mag_g)

                x_ready, _, _ = standardise_matrix(matrix_obs, mode=args.norm_mode)
                final_matrices.append(x_ready)

            final_matrices = np.asarray(final_matrices, dtype=np.float32)
            np.save(os.path.join(out_obs_dir, f"obs_matrices_{snap_stem}.npy"), final_matrices)


if __name__ == "__main__":
    main()
