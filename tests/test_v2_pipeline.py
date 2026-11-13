"""Comprehensive test and end-to-end smoke suite for Study 2 V2 (Gaia DR3 pipeline).

Tests cover:
1. Kroupa IMF + RGB luminosity sampling and Gaia DR3 G <= 20.7 depth selection.
2. Linear signed parallax space preserving negative noisy parallaxes without 1/varpi spikes.
3. Circular S^1 sky centering across RA=0/360 and leakage-free uncertainty masking.
4. N-invariant sky surface density normalization under subsampling.
5. GAT_V2 forward/backward + variance-preserving 5D -> 4D channel adaptation (even 4obs_no_RA).
6. Automatic Step 6 Best-4D log parser and strict completeness guard.
7. Full /tmp end-to-end smoke run: synthetic SWIFT HDF5 -> v2_extract_observables ->
   v2_build_graph_dataset -> v2_train_gat (Clean 5D -> Noisy 5D -> Noisy 4D) ->
   v2_infer_gaia_dr3 on real Ibata (2024) Gaia DR3 stream rows.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gaia_v2 import (
    apply_gaia_selection_and_noise_v2,
    mask_uncertainty_features,
    sample_cluster_abs_magnitudes_g,
    sample_mw_field_contaminants,
    standardise_matrix_v2,
)
from gnn_models_v2 import (
    GAT_V2,
    compute_node_relational_features_v2,
    load_pretrained_with_channel_adaptation_v2,
)
from v2_build_graph_dataset import build_sky_knn_edges
from v2_infer_gaia_dr3 import run_inference
from v2_select_best_4d import select_best_4d_config, write_selection_outputs


def test_kroupa_imf_and_gaia_limit() -> None:
    rng = np.random.default_rng(7)
    m_g = sample_cluster_abs_magnitudes_g(4000, rng=rng)
    assert m_g.shape == (4000,)
    assert np.all(np.isfinite(m_g))
    assert -0.9 <= float(np.min(m_g)) <= 2.0
    assert float(np.max(m_g)) <= 10.9

    near = np.column_stack(
        (
            rng.uniform(120.0, 122.0, 2000),
            rng.uniform(10.0, 12.0, 2000),
            np.full(2000, 6.0),
            rng.normal(1.0, 0.2, 2000),
            rng.normal(-2.0, 0.2, 2000),
            rng.normal(30.0, 4.0, 2000),
        )
    )
    far = near.copy()
    far[:, 2] = 45.0

    _, _, _, _, meta_near = apply_gaia_selection_and_noise_v2(
        near, abs_mag_g=m_g[:2000], g_max=20.7, add_noise=False, rng=rng
    )
    _, _, _, _, meta_far = apply_gaia_selection_and_noise_v2(
        far, abs_mag_g=m_g[:2000], g_max=20.7, add_noise=False, rng=rng
    )
    assert meta_near["raw_vis_frac"] > meta_far["raw_vis_frac"] * 2.5


def test_linear_parallax_preserves_negative_noisy_values() -> None:
    rng = np.random.default_rng(19)
    n = 1500
    obs_6d = np.column_stack(
        (
            rng.normal(180.0, 0.5, n),
            rng.normal(25.0, 0.5, n),
            np.full(n, 40.0),  # plx_true = 0.025 mas
            rng.normal(0.0, 0.2, n),
            rng.normal(0.0, 0.2, n),
            rng.normal(0.0, 5.0, n),
        )
    )
    faint_mg = np.full(n, 2.3)  # G ~ 20.3 mag at 40 kpc -> large plx error ~ 0.65 mas
    obs_plx, err_feats, vis_idx, _, _ = apply_gaia_selection_and_noise_v2(
        obs_6d, abs_mag_g=faint_mg, g_max=20.7, add_noise=True, rng=rng
    )
    assert len(vis_idx) == n
    assert np.all(np.isfinite(obs_plx))
    assert np.all(np.isfinite(err_feats))
    neg_mask = obs_plx[:, 2] < 0.0
    assert int(np.sum(neg_mask)) > 200
    assert np.all(err_feats[neg_mask, 3] < 0.0)


def test_circular_ra_wrap_and_uncertainty_masking() -> None:
    rng = np.random.default_rng(31)
    ra_wrap = np.concatenate((rng.uniform(358.5, 360.0, 100), rng.uniform(0.0, 1.5, 100)))
    mat_plx = np.column_stack(
        (
            ra_wrap,
            rng.normal(8.0, 0.4, 200),
            rng.normal(0.1, 0.02, 200),
            rng.normal(-0.5, 0.2, 200),
            rng.normal(1.4, 0.2, 200),
            rng.normal(-110.0, 5.0, 200),
        )
    )
    std_x, _ = standardise_matrix_v2(mat_plx)
    assert float(np.max(np.abs(std_x[:, 0]))) < 2.0  # Wrapped properly, not ~90 deg

    err = np.ones((10, 4), dtype=np.float32)
    m_no_plx = mask_uncertainty_features(err, [0, 1, 3, 4])
    assert np.allclose(m_no_plx[:, [0, 3]], 0.0) and np.allclose(m_no_plx[:, [1, 2]], 1.0)
    m_no_pmra = mask_uncertainty_features(err, [0, 1, 2, 4])
    assert np.allclose(m_no_pmra[:, 1], 0.0) and np.allclose(m_no_pmra[:, [0, 2, 3]], 1.0)


def test_n_invariant_sky_density_under_subsampling() -> None:
    rng = np.random.default_rng(101)
    r = np.sqrt(rng.uniform(0.0, 1.0, 1200))
    phi = rng.uniform(0.0, 2.0 * np.pi, 1200)
    sky_full = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
    x_full = np.hstack((sky_full, rng.normal(0.0, 1.0, (1200, 3)))).astype(np.float32)

    sub_idx = rng.choice(1200, size=300, replace=False)
    sky_sub = sky_full[sub_idx]
    x_sub = x_full[sub_idx]

    ei_f, ea_f = build_sky_knn_edges(sky_full, k_neighbors=24)
    ei_s, ea_s = build_sky_knn_edges(sky_sub, k_neighbors=24)

    rel_f = compute_node_relational_features_v2(
        torch.from_numpy(x_full), torch.from_numpy(ei_f), edge_attr=torch.from_numpy(ea_f)
    )
    rel_s = compute_node_relational_features_v2(
        torch.from_numpy(x_sub), torch.from_numpy(ei_s), edge_attr=torch.from_numpy(ea_s)
    )

    rho_col = 3 * 5 + 1
    med_f = float(torch.median(rel_f[:, rho_col]))
    med_s = float(torch.median(rel_s[:, rho_col]))
    assert abs(med_f - med_s) < 0.05


def test_gat_v2_and_5d_to_4d_transfer() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(22)
    sky = rng.normal(0.0, 1.0, (60, 2))
    ei, ea = build_sky_knn_edges(sky, k_neighbors=12)

    x_5d = torch.randn(60, 5)
    err_x = torch.rand(60, 4)
    m5 = GAT_V2(in_features=5, err_features=4, hidden_channels=32, heads=4, out_channels=3, edge_dim=3)
    out = m5(x_5d, torch.from_numpy(ei), edge_attr=torch.from_numpy(ea), err_x=err_x)
    assert out.shape == (60, 3)
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m5.parameters())

    # Transfer into 4obs_no_RA (drops col 0) AND verify 4D with edge_attr runs cleanly
    keep = [1, 2, 3, 4]
    m4 = GAT_V2(in_features=4, err_features=4, hidden_channels=32, heads=4, out_channels=3, edge_dim=3)
    loaded, skipped = load_pretrained_with_channel_adaptation_v2(
        m4, m5.state_dict(), selected_features=keep, d_err=4
    )
    assert skipped == 0 and loaded > 0
    out4 = m4(x_5d[:, keep], torch.from_numpy(ei), edge_attr=torch.from_numpy(ea), err_x=err_x)
    assert out4.shape == (60, 3) and torch.all(torch.isfinite(out4))


def test_best_4d_selector_logic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mock_scores = {
            "4obs_no_RA": (0.28, 86.0, 84.0, 0.78, 9.5),
            "4obs_no_Dec": (0.27, 87.0, 85.0, 0.80, 8.2),
            "4obs_no_Parallax": (0.21, 91.5, 90.2, 0.88, 3.8),
            "4obs_no_pmRA": (0.35, 80.0, 77.0, 0.68, 15.0),
            "4obs_no_pmDec": (0.33, 81.5, 79.0, 0.71, 13.0),
        }
        for tag, (loss, acc, bacc, f1, cfnr) in mock_scores.items():
            d = os.path.join(tmp, f"{tag}_noisy")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"training_metrics_gat_{tag}_noisy.log"), "w", encoding="utf-8") as f:
                f.write(
                    f"Final Test Loss : {loss:.4f} | Acc: {acc:.2f}% | BalAcc: {bacc:.2f}% | "
                    f"F1: {f1:.4f} | Prec: {f1:.4f} | Rec: {f1:.4f} | "
                    f"CoreFNR: {cfnr:.2f}% | CoreRec: {100.0 - cfnr:.2f}%\n"
                )

        sel = select_best_4d_config(tmp, require_all=True)
        assert sel["best_tag"] == "4obs_no_Parallax"
        assert sel["feature_indices"] == [0, 1, 3, 4]
        out_cfg = os.path.join(tmp, "out_cfg")
        write_selection_outputs(sel, out_cfg)
        assert os.path.isfile(os.path.join(out_cfg, "best_4d_config.json"))
        assert os.path.isfile(os.path.join(out_cfg, "best_4d_config.env"))

        # Remove one log and assert strict completeness fails unless allow_partial
        os.remove(os.path.join(tmp, "4obs_no_RA_noisy", "training_metrics_gat_4obs_no_RA_noisy.log"))
        raised = False
        try:
            select_best_4d_config(tmp, require_all=True)
        except RuntimeError:
            raised = True
        assert raised


def _write_synthetic_swift_hdf5(path: str, seed: int, n_part: int = 400, box_shift: float = 500.0) -> None:
    rng = np.random.default_rng(seed)
    # Tight King/Plummer-like bound core centered on halo orbit (12, 4, 8) kpc + unbound tidal stars
    n_core = int(0.65 * n_part)
    n_tail = n_part - n_core
    r_core = rng.normal(0.0, 0.012, (n_core, 3))
    v_core = rng.normal(0.0, 2.5, (n_core, 3))
    r_tail = rng.normal(0.0, 0.65, (n_tail, 3))
    v_tail = rng.normal(0.0, 45.0, (n_tail, 3))

    center = np.array([12.0, 4.0, 8.0])
    v_bulk = np.array([-85.0, 130.0, 45.0])
    pos = np.vstack((r_core, r_tail)) + center
    vel = np.vstack((v_core, v_tail)) + v_bulk
    masses = np.full(n_part, 20.0, dtype=np.float64)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        dm = f.create_group("DMParticles")
        dm.create_dataset("Coordinates", data=pos + box_shift)
        dm.create_dataset("Velocities", data=vel)
        dm.create_dataset("Masses", data=masses)
        hdr = f.create_group("Header")
        hdr.attrs["Time"] = np.array([1.0], dtype=np.float64)
        units = f.create_group("Units")
        units.attrs["Unit time in cgs (U_t)"] = np.array([31557600.0e9], dtype=np.float64)


def test_end_to_end_smoke_pipeline() -> None:
    py = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        sim_dir = os.path.join(tmp, "outputs_unbiased")
        obs_dir = os.path.join(tmp, "obs_v2")
        g_clean = os.path.join(tmp, "graphs_clean")
        g_noisy = os.path.join(tmp, "graphs_noisy")
        models_dir = os.path.join(tmp, "models")

        for ic in range(3):
            for s in range(2):
                _write_synthetic_swift_hdf5(
                    os.path.join(sim_dir, f"IC_{ic}", f"snapshot_000{s}.hdf5"),
                    seed=100 * ic + s,
                )

        subprocess.run(
            [
                py,
                os.path.join(ROOT_DIR, "scripts", "v2_extract_observables.py"),
                "--sim_dir", sim_dir,
                "--out_dir", obs_dir,
                "--n_rotations", "2",
                "--contaminant_min", "0.15",
                "--contaminant_max", "0.35",
                "--num_workers", "1",
            ],
            check=True,
        )

        for regime, g_dir in (("clean", g_clean), ("noisy", g_noisy)):
            subprocess.run(
                [
                    py,
                    os.path.join(ROOT_DIR, "scripts", "v2_build_graph_dataset.py"),
                    "--obs_dir", obs_dir,
                    "--output_dir", g_dir,
                    "--regime", regime,
                    "--k_neighbors", "16",
                    "--num_workers", "1",
                ],
                check=True,
            )

        w_c5 = os.path.join(models_dir, "clean_5d.pth")
        w_n5 = os.path.join(models_dir, "noisy_5d.pth")
        w_n4 = os.path.join(models_dir, "noisy_4d.pth")

        # Stage 1: Clean 5D
        subprocess.run(
            [
                py,
                os.path.join(ROOT_DIR, "scripts", "v2_train_gat.py"),
                "--data_folder", g_clean,
                "--feature_indices", "0,1,2,3,4",
                "--bs", "4",
                "--micro_bs", "2",
                "--epochs", "2",
                "--save_model", w_c5,
                "--log_file", os.path.join(models_dir, "clean_5d.log"),
                "--no_resume",
            ],
            check=True,
        )
        assert os.path.isfile(w_c5)

        # Stage 2: Clean 5D -> Noisy 5D
        subprocess.run(
            [
                py,
                os.path.join(ROOT_DIR, "scripts", "v2_train_gat.py"),
                "--data_folder", g_noisy,
                "--feature_indices", "0,1,2,3,4",
                "--bs", "4",
                "--micro_bs", "2",
                "--epochs", "2",
                "--pretrained_weights", w_c5,
                "--save_model", w_n5,
                "--log_file", os.path.join(models_dir, "noisy_5d.log"),
                "--no_resume",
            ],
            check=True,
        )
        assert os.path.isfile(w_n5)

        # Stage 3: 5D -> Best-4D transfer
        subprocess.run(
            [
                py,
                os.path.join(ROOT_DIR, "scripts", "v2_train_gat.py"),
                "--data_folder", g_noisy,
                "--feature_indices", "0,1,3,4",
                "--bs", "4",
                "--micro_bs", "2",
                "--epochs", "2",
                "--pretrained_weights", w_n5,
                "--save_model", w_n4,
                "--log_file", os.path.join(models_dir, "noisy_4d.log"),
                "--no_resume",
            ],
            check=True,
        )
        assert os.path.isfile(w_n4)

        # Inference test on real Ibata (2024) CSV subset (Streams 1 and 3, including RA wrap & 250 kpc outliers)
        ibata_sub = os.path.join(tmp, "ibata_subset.csv")
        src_ibata = os.path.join(ROOT_DIR, "make_ICs", "stars_6D_ibata2024.csv")
        with open(src_ibata, "r", encoding="utf-8") as fin, open(ibata_sub, "w", encoding="utf-8", newline="") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for r in reader:
                if r["Stream_ID"] in ("1", "2", "3", "4", "5", "6"):
                    writer.writerow(r)

        out_csv = os.path.join(tmp, "ibata_pred.csv")
        out_sum = os.path.join(tmp, "ibata_summary.json")
        summary = run_inference(
            input_csv=ibata_sub,
            out_csv=out_csv,
            out_summary=out_sum,
            model_5d_path=w_n5,
            model_4d_path=w_n4,
            mode="auto",
            k_neighbors=16,
        )
        assert summary["groups_evaluated"] >= 2
        with open(out_csv, "r", encoding="utf-8") as f:
            pred_rows = list(csv.DictReader(f))
        assert len(pred_rows) >= 10
        evaluated_count = 0
        for pr in pred_rows:
            if pr["Pred_Label"] == "Skipped":
                continue
            evaluated_count += 1
            probs = [float(pr["P_Unbound_Field"]), float(pr["P_Potential_Escaper"]), float(pr["P_Bound_Core"])]
            assert abs(sum(probs) - 1.0) < 1e-4
        assert evaluated_count >= 10


if __name__ == "__main__":
    test_kroupa_imf_and_gaia_limit()
    test_linear_parallax_preserves_negative_noisy_values()
    test_circular_ra_wrap_and_uncertainty_masking()
    test_n_invariant_sky_density_under_subsampling()
    test_gat_v2_and_5d_to_4d_transfer()
    test_best_4d_selector_logic()
    test_end_to_end_smoke_pipeline()
    print("ALL 7 STUDY 2 V2 TESTS PASSED CLEANLY.")
