"""Generate SWIFT Plummer initial conditions (.hdf5) for the unbiased orbit sample.

Reads `make_ICs/SWIFT_seeds_unbiased.npy` (produced by `select_unbiased_orbits.py`)
and generates isotropic virialized Plummer spheres in `make_ICs/initial_conditions_unbiased/`,
keeping the baseline study's `make_ICs/initial_conditions/` directory untouched.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

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
    nb_threads=4,
    seed=42,
):
    """Generate Plummer HDF5 initial conditions for all selected unbiased progenitor seeds."""
    if seeds_path is None:
        seeds_path = os.path.join(THIS_DIR, "SWIFT_seeds_unbiased.npy")
    if output_dir is None:
        output_dir = os.path.join(THIS_DIR, "initial_conditions_unbiased")

    os.makedirs(output_dir, exist_ok=True)
    seeds = np.load(seeds_path, allow_pickle=True)

    df_seeds = pd.DataFrame(
        {
            "Stream_ID": seeds[:, 0].astype(int),
            "x_kpc": seeds[:, 1],
            "y_kpc": seeds[:, 2],
            "z_kpc": seeds[:, 3],
            "vx_kms": seeds[:, 4],
            "vy_kms": seeds[:, 5],
            "vz_kms": seeds[:, 6],
        }
    )

    rng = np.random.default_rng(seed)
    n_streams = len(df_seeds)
    particle_counts = rng.integers(min_particles, max_particles + 1, size=n_streams)
    scale_radii = rng.uniform(min_scale_kpc, max_scale_kpc, size=n_streams)

    ic_metadata = []
    for idx, row in enumerate(df_seeds.itertuples(index=False)):
        sid = int(row.Stream_ID)
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
                "x_init_kpc": row.x_kpc,
                "y_init_kpc": row.y_kpc,
                "z_init_kpc": row.z_kpc,
                "vx_init_kms": row.vx_kms,
                "vy_init_kms": row.vy_kms,
                "vz_init_kms": row.vz_kms,
                "hdf5_path": fname,
            }
        )

        if not os.path.exists(fname):
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
                shiftx=row.x_kpc,
                shifty=row.y_kpc,
                shiftz=row.z_kpc,
                shiftvx=row.vx_kms,
                shiftvy=row.vy_kms,
                shiftvz=row.vz_kms,
                fname=fname,
            )

    meta_csv = os.path.join(output_dir, "unbiased_plummer_parameters.csv")
    pd.DataFrame(ic_metadata).to_csv(meta_csv, index=False)
    print(f"Saved Plummer IC metadata table to: {meta_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Plummer HDF5 initial conditions for the unbiased orbit sample."
    )
    parser.add_argument(
        "--seeds_path",
        type=str,
        default=None,
        help="Path to SWIFT_seeds_unbiased.npy.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for HDF5 files (default: make_ICs/initial_conditions_unbiased).",
    )
    parser.add_argument(
        "--nb_threads",
        type=int,
        default=4,
        help="Number of worker threads for velocity sampling (default: 4).",
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
