################################################################################
# Copyright (c) 2022 Patrick Hirling (patrick.hirling@epfl.ch)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Adapted by Adrian Cozma 2026, adrian.cozma@epfl.ch
#
################################################################################

import numpy as np
import scipy.special as sci
from scipy.optimize import minimize
import time
import argparse
try:
    from swiftsimio import Writer
    import unyt
    _HAS_SWIFTSIMIO = True
except ImportError:
    import h5py
    _HAS_SWIFTSIMIO = False


def _write_swift_hdf5_h5py(fname, X_kpc, V_kms, m_msun, boxsize_kpc):
    """Write a SWIFT-compatible HDF5 initial conditions file via h5py.

    Matches the exact `/Header`, `/Units`, and `/PartType1` schema and CGS
    conversion attributes produced by `swiftsimio.Writer(galactic_units)`.
    """
    n_part = len(m_msun)
    u_l_cgs = 3.08567758e21       # 1 kpc in cm
    u_m_cgs = 1.98841586e43       # 1e10 M_sun in g
    u_t_cgs = 3.08567758e19       # 1 Mpc / (km/s) in s (vel unit = 1e-3 km/s = 100 cm/s)

    with h5py.File(fname, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["BoxSize"] = float(boxsize_kpc)
        hdr.attrs["Dimension"] = np.array([3], dtype=np.int64)
        hdr.attrs["Flag_Entropy_ICs"] = np.int64(0)
        hdr.attrs["MassTable"] = np.array([0.0, float(m_msun[0]) * 1e-10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        hdr.attrs["NumFilesPerSnapshot"] = np.int64(1)
        hdr.attrs["NumPart_ThisFile"] = np.array([0, n_part, 0, 0, 0, 0, 0], dtype=np.int64)
        hdr.attrs["NumPart_Total"] = np.array([0, n_part, 0, 0, 0, 0, 0], dtype=np.int64)
        hdr.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.int64)

        units = f.create_group("Units")
        units.attrs["Unit current in cgs (U_I)"] = np.array([1.0], dtype=np.float64)
        units.attrs["Unit length in cgs (U_L)"] = np.array([u_l_cgs], dtype=np.float64)
        units.attrs["Unit mass in cgs (U_M)"] = np.array([u_m_cgs], dtype=np.float64)
        units.attrs["Unit temperature in cgs (U_T)"] = np.array([1.0], dtype=np.float64)
        units.attrs["Unit time in cgs (U_t)"] = np.array([u_t_cgs], dtype=np.float64)

        pt1 = f.create_group("PartType1")

        def _set_unit_attrs(dset, conv_cgs, u_l=0.0, u_m=0.0, u_t=0.0):
            dset.attrs["Conversion factor to CGS (not including cosmological corrections)"] = np.array([conv_cgs], dtype=np.float64)
            dset.attrs["Conversion factor to physical CGS (including cosmological corrections)"] = np.array([conv_cgs], dtype=np.float64)
            dset.attrs["U_I exponent"] = np.array([0.0], dtype=np.float64)
            dset.attrs["U_L exponent"] = np.array([u_l], dtype=np.float64)
            dset.attrs["U_M exponent"] = np.array([u_m], dtype=np.float64)
            dset.attrs["U_T exponent"] = np.array([0.0], dtype=np.float64)
            dset.attrs["U_t exponent"] = np.array([u_t], dtype=np.float64)
            dset.attrs["a-scale exponent"] = np.array([0], dtype=np.int64)
            dset.attrs["h-scale exponent"] = np.array([0.0], dtype=np.float64)

        d_coords = pt1.create_dataset("Coordinates", data=np.asarray(X_kpc, dtype=np.float64))
        _set_unit_attrs(d_coords, u_l_cgs, u_l=1.0)

        d_vels = pt1.create_dataset("Velocities", data=np.asarray(V_kms, dtype=np.float64) * 1000.0)
        _set_unit_attrs(d_vels, 100.0, u_l=1.0, u_t=-1.0)

        d_masses = pt1.create_dataset("Masses", data=np.asarray(m_msun, dtype=np.float64) * 1e-10)
        _set_unit_attrs(d_masses, u_m_cgs, u_m=1.0)

        d_ids = pt1.create_dataset("ParticleIDs", data=np.arange(n_part, dtype=np.int64))
        _set_unit_attrs(d_ids, 1.0)

_PLUMMER_CTX = {}


def _sample_vel_worker(r):
    sample_fn = _PLUMMER_CTX["sample_vel"]
    return sample_fn(r)


def generate_plummer(a=0.01, M=1e5, q=1.0, N=10000, bound=2.0, boxsize=4.0, 
                     use_parallel=True, nb_threads=2, 
                     shiftx=0.0, shifty=0.0, shiftz=0.0, 
                     shiftvx=0.0, shiftvy=0.0, shiftvz=0.0, 
                     fname="plummer.hdf5"):

    if q > 2.0:
        raise ValueError(f"Anisotropy parameter must be smaller than 2.0 (q = {q:.2f} given)")

    #### Parameters
    # Plummer Model
    G = 4.299581e-06  # Gravitational constant [kpc / M_s * (kms)^2]
    N = int(N)
    pickle_ics = 0  
    periodic_bdry = 0

    ### Print parameters
    print("\n############################################################################################# \n")
    print("Generating Plummer ICs with the following parameters::\n")
    print(f"Softening length:                                  a = {a:.3e} kpc")
    print(f"Total Mass:                                        M = {M:.3e} Solar Masses")
    print(f"Anisotropy parameter:                              q = {q}")
    print(f"Output file:                                       {fname}")
    print(f"Number of particles:                               N = {N:.1e}")
    print("\n############################################################################################# \n")

    # Estimate good softening length
    epsilon0 = a * N ** (-1.0 / 3.0)
    print(f"Recommended Softening length: {epsilon0:.4e} kpc")

    ### Generate Positions
    def r_of_m(m_val, a_val, M_val):
        return a_val * ((M_val / m_val) ** (2.0 / 3.0) - 1) ** (-1.0 / 2.0)

    m_rand = M * np.random.uniform(0.0, 1.0, N)
    r_rand = r_of_m(m_rand, a, M)
    phi_rand = np.random.uniform(0.0, 2 * np.pi, N)
    theta_rand = np.arccos(np.random.uniform(-1.0, 1.0, N))

    x = r_rand * np.sin(theta_rand) * np.cos(phi_rand)
    y = r_rand * np.sin(theta_rand) * np.sin(phi_rand)
    z = r_rand * np.cos(theta_rand)

    X = np.array([x, y, z]).transpose()

    ### Generate Velocities
    normalization = 3.0 * sci.gamma(6.0 - q) / (2.0 * (2.0 * np.pi) ** (2.5))
    units = G ** (q - 5.0) * M ** (q - 4.0) * a ** (2.0 - q)

    def H(E, L):
        x_val = L ** 2 / (2.0 * E * a ** 2)
        if x_val <= 1.0:
            return 1.0 / (sci.gamma(4.5 - q)) * sci.hyp2f1(0.5 * q, q - 3.5, 1.0, x_val)
        else:
            return (
                1.0 / (sci.gamma(1.0 - 0.5 * q) * sci.gamma(4.5 - 0.5 * q) * (x_val ** (0.5 * q)))
                * sci.hyp2f1(0.5 * q, 0.5 * q, 4.5 - 0.5 * q, 1.0 / x_val)
            )

    def Fq(E, L):
        if E < 0.0:
            return 0.0
        else:
            return normalization * units * E ** (3.5 - q) * H(E, L)

    def relative_potential(r): 
        return G * M / np.sqrt(r ** 2 + a ** 2)

    def relative_energy(r, vr, vt):
        return relative_potential(r) - 0.5 * (vr ** 2 + vt ** 2)

    def Fq_tomin(v, r):
        return -Fq(relative_energy(r, v[0], v[1]), r * v[1]) * v[1]

    def fmax(r, vmax):
        def vel_constr2(v):
            return vmax ** 2 - v[0] ** 2 - v[1] ** 2
        v0 = [0.1 * vmax, 0.2 * vmax]
        cons = {"type": "ineq", "fun": vel_constr2}
        args_tuple = (r,)
        fm = minimize(
            Fq_tomin, v0, constraints=cons, method="SLSQP", args=args_tuple,
            bounds=[(0, vmax), (0, vmax)]
        )
        return -fm.fun

    def sample_vel(r):
        vmax = np.sqrt(2.0 * relative_potential(r))
        fm = 1.1 * fmax(r, vmax) 
        while True:
            while True:
                vr = np.random.uniform(0.0, vmax)
                vt = np.random.uniform(0.0, vmax)
                if vr ** 2 + vt ** 2 <= vmax ** 2:
                    break
            f = np.random.uniform(0.0, fm)
            if Fq(relative_energy(r, vr, vt), r * vt) * vt >= f:
                return vr, vt

    print("Sampling velocities...")
    ti = time.time()
    if use_parallel:
        import multiprocessing as mp
        _PLUMMER_CTX["sample_vel"] = sample_vel
        ctx = mp.get_context("fork")
        with ctx.Pool(nb_threads) as p:
            vels = np.array(p.map(_sample_vel_worker, r_rand, chunksize=max(1, N // (nb_threads * 4)))).transpose()
    else:
        from tqdm import tqdm
        vels = np.empty((2, N))
        for j, r in enumerate(tqdm(r_rand)):
            vels[:, j] = sample_vel(r)
    tf = time.time()
    print(f"Sampling took {tf - ti:.2f} seconds.")

    alph = np.random.uniform(0, 2 * np.pi, N)
    sgn = np.random.choice([-1, 1], size=N)
    vphi = vels[1] * np.cos(alph)
    vtheta = vels[1] * np.sin(alph)
    vr = sgn * vels[0]

    v_x = (np.sin(theta_rand) * np.cos(phi_rand) * vr + np.cos(theta_rand) * np.cos(phi_rand) * vtheta - np.sin(phi_rand) * vphi)
    v_y = (np.sin(theta_rand) * np.sin(phi_rand) * vr + np.cos(theta_rand) * np.sin(phi_rand) * vtheta + np.cos(phi_rand) * vphi)
    v_z = np.cos(theta_rand) * vr - np.sin(theta_rand) * vtheta

    V = np.array([v_x, v_y, v_z]).transpose()
    m = M / N * np.ones(N)

    ### Exclude extreme outliers
    idx = r_rand < bound
    X = X[idx]
    V = V[idx]
    m = m[idx]
    new_N = len(m)

    # Boost into orbit
    V += np.array([shiftvx, shiftvy, shiftvz])
    X += np.array([shiftx, shifty, shiftz])

    ### Write to hdf5
    if _HAS_SWIFTSIMIO:
        galactic_units = unyt.UnitSystem(
            "galactic", unyt.kpc, unyt.unyt_quantity(1e10, units=unyt.Solar_Mass),
            unyt.unyt_quantity(1.0, units=unyt.s * unyt.Mpc / unyt.km).to(unyt.Gyr),
        )
        wrt = Writer(galactic_units, boxsize * unyt.kpc)
        wrt.dark_matter.coordinates = X * unyt.kpc
        wrt.dark_matter.velocities = V * (unyt.km / unyt.s)
        wrt.dark_matter.masses = m * unyt.msun
        wrt.dark_matter.particle_ids = np.arange(new_N)
        wrt.write(fname)
    else:
        _write_swift_hdf5_h5py(fname, X, V, m, boxsize)
    print("Writing IC file...")
    
    return fname 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate discrete realization of Anisotropic Plummer Model")
    parser.add_argument("-a", type=float, default=0.05)
    parser.add_argument("-M", type=float, default=1.0e5)
    parser.add_argument("-q", type=float, default=1.0)
    parser.add_argument("-N", type=int, default=10000)
    parser.add_argument("-bound", type=float, default=2.0)
    parser.add_argument("-boxsize", type=float, default=4.0)
    parser.add_argument("--noparallel", action="store_false", dest="use_parallel")
    parser.add_argument("-nbthreads", type=int, default=2)
    parser.add_argument("-shiftx", type=float, default=0.0)
    parser.add_argument("-shifty", type=float, default=0.0)
    parser.add_argument("-shiftz", type=float, default=0.0)
    parser.add_argument("-shiftvx", type=float, default=0.0)
    parser.add_argument("-shiftvy", type=float, default=0.0)
    parser.add_argument("-shiftvz", type=float, default=0.0)
    parser.add_argument("-fname", type=str, default="plummer.hdf5")

    args = parser.parse_args()

    generate_plummer(a=args.a, M=args.M, q=args.q, N=args.N, bound=args.bound, boxsize=args.boxsize, 
                     use_parallel=args.use_parallel, nbthreads=args.nbthreads, 
                     shiftx=args.shiftx, shifty=args.shifty, shiftz=args.shiftz, 
                     shiftvx=args.shiftvx, shiftvy=args.shiftvy, shiftvz=args.shiftvz, 
                     fname=args.fname)