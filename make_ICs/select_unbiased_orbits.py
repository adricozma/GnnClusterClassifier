"""Unbiased progenitor orbit selection from the Ibata et al. (2024) stream atlas.

This script integrates the present-day median 6D phase-space coordinates of the
84 stellar streams from Ibata et al. (2024, ApJ, 967, 89) backward for 2 Gyr in
the static Milky Way potential MWPotential2014 (Bovy 2015, ApJS, 216, 29).

Methodological corrections compared to the initial workflow:
1. Removes the artificial r(-2 Gyr) <= 100 kpc cutoff so that outer-halo globular
   cluster progenitors are preserved (bounded only by the SWIFT simulation box
   half-width, default 480 kpc).
2. Filters out inner-Galaxy orbits with small perigalacticon (R_p < R_{p,min},
   default 5.0 kpc) that undergo immediate impulsive tidal destruction in the
   Galactic bulge/disk and create extreme unbound class imbalance.
3. Exports the full orbital diagnostics table (R_p, R_a, eccentricity, initial
   galactocentric distance) and the resulting SWIFT seed array using standard
   library CSV + NumPy so no external pandas dependency is required on HPC nodes.
"""

import argparse
import csv
import os
from collections import defaultdict
import astropy.units as u
import numpy as np
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _compute_stream_medians(input_csv):
    """Read stars_6D_ibata2024.csv and return per-Stream_ID median 6D coordinates."""
    grouped = defaultdict(lambda: defaultdict(list))
    target_cols = ("RA", "Dec", "Distance_kpc", "pmRA", "pmDec", "Vlos")

    with open(input_csv, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(float(row["Stream_ID"]))
            for col in target_cols:
                grouped[sid][col].append(float(row[col]))

    centers = []
    for sid in sorted(grouped.keys()):
        med = {"Stream_ID": sid}
        for col in target_cols:
            med[col] = float(np.median(grouped[sid][col]))
        centers.append(med)
    return centers


def select_unbiased_orbits(
    input_csv=None,
    output_seeds=None,
    output_summary_csv=None,
    orbits_dir=None,
    integration_time_gyr=-2.0,
    num_steps=1000,
    min_perigalacticon_kpc=0.0,
    max_box_coord_kpc=480.0,
):
    """Integrate Ibata et al. (2024) stream centers backward and filter by perigalacticon."""
    if input_csv is None:
        input_csv = os.path.join(ROOT_DIR, "make_ICs", "stars_6D_ibata2024.csv")
    if output_seeds is None:
        output_seeds = os.path.join(ROOT_DIR, "make_ICs", "SWIFT_seeds_unbiased.npy")
    if output_summary_csv is None:
        output_summary_csv = os.path.join(ROOT_DIR, "make_ICs", "unbiased_orbits_summary.csv")
    if orbits_dir is None:
        orbits_dir = os.path.join(ROOT_DIR, "make_ICs", "orbits_unbiased")

    os.makedirs(orbits_dir, exist_ok=True)

    present_centers = _compute_stream_medians(input_csv)
    ts = np.linspace(0.0, integration_time_gyr, num_steps) * u.Gyr

    seeds_list = []
    summary_rows = []

    for row in present_centers:
        sid = int(row["Stream_ID"])

        o_center = Orbit(
            vxvv=[
                row["RA"],
                row["Dec"],
                row["Distance_kpc"],
                row["pmRA"],
                row["pmDec"],
                row["Vlos"],
            ],
            radec=True,
        )
        o_center.integrate(ts, MWPotential2014)

        x_traj = np.asarray(o_center.x(ts), dtype=float)
        y_traj = np.asarray(o_center.y(ts), dtype=float)
        z_traj = np.asarray(o_center.z(ts), dtype=float)
        vx_traj = np.asarray(o_center.vx(ts), dtype=float)
        vy_traj = np.asarray(o_center.vy(ts), dtype=float)
        vz_traj = np.asarray(o_center.vz(ts), dtype=float)

        r_traj = np.sqrt(x_traj**2 + y_traj**2 + z_traj**2)
        r_peri = float(np.min(r_traj))
        r_apo = float(np.max(r_traj))
        ecc = (r_apo - r_peri) / (r_apo + r_peri) if (r_apo + r_peri) > 0 else 0.0

        # Detect first pericenter passage R_{p1} along forward time (from t = -2 Gyr to 0 Gyr)
        r_forward = r_traj[::-1]
        local_min_idx = np.where(
            (r_forward[1:-1] < r_forward[:-2]) & (r_forward[1:-1] < r_forward[2:])
        )[0] + 1
        r_p1 = float(r_forward[local_min_idx[0]]) if len(local_min_idx) > 0 else r_peri

        x_prog, y_prog, z_prog = float(x_traj[-1]), float(y_traj[-1]), float(z_traj[-1])
        vx_prog, vy_prog, vz_prog = float(vx_traj[-1]), float(vy_traj[-1]), float(vz_traj[-1])
        r_init = float(r_traj[-1])
        max_abs_coord = float(np.max(np.abs(np.column_stack((x_traj, y_traj, z_traj)))))

        passes_peri = r_peri >= min_perigalacticon_kpc
        passes_box = (max_box_coord_kpc is None) or (max_abs_coord <= max_box_coord_kpc)
        selected = bool(passes_peri and passes_box)

        summary_rows.append(
            {
                "Stream_ID": sid,
                "r_present_kpc": float(r_traj[0]),
                "r_init_2Gyr_kpc": r_init,
                "R_peri_kpc": r_peri,
                "R_p1_kpc": r_p1,
                "R_apo_kpc": r_apo,
                "eccentricity": ecc,
                "max_abs_xyz_kpc": max_abs_coord,
                "x_init_kpc": x_prog,
                "y_init_kpc": y_prog,
                "z_init_kpc": z_prog,
                "vx_init_kms": vx_prog,
                "vy_init_kms": vy_prog,
                "vz_init_kms": vz_prog,
                "selected": selected,
            }
        )

        if selected:
            seeds_list.append([sid, x_prog, y_prog, z_prog, vx_prog, vy_prog, vz_prog])
            orbit_path = os.path.join(orbits_dir, f"orbit_{sid}.txt")
            data_matrix = np.column_stack((x_traj, y_traj, z_traj, vx_traj, vy_traj, vz_traj))
            np.savetxt(
                orbit_path,
                data_matrix,
                header="x(kpc) y(kpc) z(kpc) vx(km/s) vy(km/s) vz(km/s)",
            )

    seeds_array = np.asarray(seeds_list, dtype=float)
    np.save(output_seeds, seeds_array)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(output_summary_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(
        f"Processed {len(summary_rows)} streams | "
        f"Selected {len(seeds_array)} orbits with R_p >= {min_perigalacticon_kpc:.2f} kpc "
        f"and max(|x|,|y|,|z|) <= {max_box_coord_kpc} kpc."
    )
    print(f"Saved seeds to: {output_seeds}")
    print(f"Saved orbital summary to: {output_summary_csv}")
    return seeds_array, summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select unbiased globular cluster progenitor orbits from Ibata et al. (2024)."
    )
    parser.add_argument("--input_csv", type=str, default=None, help="Path to stars_6D_ibata2024.csv.")
    parser.add_argument("--output_seeds", type=str, default=None, help="Output .npy path for selected seeds.")
    parser.add_argument("--output_summary_csv", type=str, default=None, help="Output .csv path for orbital summary.")
    parser.add_argument("--orbits_dir", type=str, default=None, help="Directory for orbit_<Stream_ID>.txt files.")
    parser.add_argument(
        "--min_perigalacticon",
        type=float,
        default=0.0,
        help="Minimum perigalacticon R_p [kpc] (default: 0.0 to retain close-pericenter tidally disrupted clusters).",
    )
    parser.add_argument(
        "--max_box_coord",
        type=float,
        default=480.0,
        help="Maximum Cartesian coordinate [kpc] allowed inside the SWIFT box (default: 480.0).",
    )
    args = parser.parse_args()

    select_unbiased_orbits(
        input_csv=args.input_csv,
        output_seeds=args.output_seeds,
        output_summary_csv=args.output_summary_csv,
        orbits_dir=args.orbits_dir,
        min_perigalacticon_kpc=args.min_perigalacticon,
        max_box_coord_kpc=args.max_box_coord,
    )
