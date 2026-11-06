"""Generate SWIFT Plummer initial conditions (.hdf5) for the unbiased orbit sample.

Reads `make_ICs/SWIFT_seeds_unbiased.npy` (produced by `select_unbiased_orbits.py`)
and generates isotropic virialized Plummer spheres in `make_ICs/initial_conditions_unbiased/`,
keeping the baseline study's `make_ICs/initial_conditions/` directory untouched.
"""

import argparse
import csv
import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from IC_plummer_swift import generate_plummer


def generate_all_unbiased_ics(
    seeds_path=None,
    output_dir=None,
    min_particles=20000,
    max_particles=50000,
    min_scale_kpc=0.001,
    max_scale_kpc=0.05,
    boxsize_kpc=1000.0,
    nb_threads=16,
    seed=42,
):
    """Generate Plummer HDF5 initial conditions for all selected unbiased progenitor seeds."""
    if seeds_path is None:
        seeds_path = os.path.join(THIS_DIR, "SWIFT_seeds_unbiased.npy")
    if output_dir is None:
        output_dir = os.path.join(THIS_DIR, "initial_conditions_unbiased")

    os.makedirs(output_dir, exist_ok=True)
    seeds = np.load(seeds_path, allow_pickle=True)

    rng = np.random.default_rng(seed)
    n_streams = len(seeds)
    particle_counts = rng.integers(min_particles, max_particles + 1, size=n_streams)
    scale_radii = rng.uniform(min_scale_kpc, max_scale_kpc, size=n_streams)

    ic_metadata = []
    for idx in range(n_streams):
        sid = int(seeds[idx, 0])
        x_kpc, y_kpc, z_kpc = float(seeds[idx, 1]), float(seeds[idx, 2]), float(seeds[idx, 3])
        vx_kms, vy_kms, vz_kms = float(seeds[idx, 4]), float(seeds[idx, 5]), float(seeds[idx, 6])
        n_part = int(particle_counts[idx])
        mass_msun = float(n_part) * 1.0
        a_kpc = float(scale_radii[idx])
        fname = os.path.join(output_dir, f"IC_{sid}.hdf5")

        ic_metadata.append(
            {
                "Stream_ID": sid,
                "N_particles": n_part,
                "Mass_Msun": mass_msun,
                "Plummer_a_kpc": a_kpc,
                "x_init_kpc": x_kpc,
                "y_init_kpc": y_kpc,
                "z_init_kpc": z_kpc,
                "vx_init_kms": vx_kms,
                "vy_init_kms": vy_kms,
                "vz_init_kms": vz_kms,
                "hdf5_path": fname,
            }
        )

        existing_study1_ic = f"/home/cozma/swift_sims/ICs/IC_{sid}.hdf5"
        if not os.path.exists(fname) and os.path.exists(existing_study1_ic):
            os.symlink(existing_study1_ic, fname)
            print(f"[{idx + 1}/{n_streams}] Linked existing Study 1 IC: {existing_study1_ic} -> {fname}")
        elif not os.path.exists(fname):
            print(f"[{idx + 1}/{n_streams}] Generating {fname} (N={n_part}, a={a_kpc:.4f} kpc)...")
            generate_plummer(
                a=a_kpc,
                M=mass_msun,
                q=1.0,
                N=n_part,
                bound=2.0,
                boxsize=boxsize_kpc,
                use_parallel=(nb_threads > 1),
                nb_threads=nb_threads,
                shiftx=x_kpc,
                shifty=y_kpc,
                shiftz=z_kpc,
                shiftvx=vx_kms,
                shiftvy=vy_kms,
                shiftvz=vz_kms,
                fname=fname,
            )

    meta_csv = os.path.join(output_dir, "unbiased_plummer_parameters.csv")
    if ic_metadata:
        fieldnames = list(ic_metadata[0].keys())
        with open(meta_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ic_metadata)
    print(f"Saved Plummer IC metadata table to: {meta_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Plummer HDF5 initial conditions for the unbiased orbit sample."
    )
    parser.add_argument("--seeds_path", type=str, default=None, help="Path to SWIFT_seeds_unbiased.npy.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for HDF5 files (default: make_ICs/initial_conditions_unbiased).",
    )
    parser.add_argument(
        "--nb_threads",
        type=int,
        default=16,
        help="Number of worker threads for velocity sampling (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible Plummer masses and scale radii.",
    )
    args = parser.parse_args()

    generate_all_unbiased_ics(
        seeds_path=args.seeds_path,
        output_dir=args.output_dir,
        nb_threads=args.nb_threads,
        seed=args.seed,
    )
