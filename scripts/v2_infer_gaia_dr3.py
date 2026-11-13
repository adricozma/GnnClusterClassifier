"""Calibrated Study 2 V2 inference pipeline for real Gaia DR3 catalogs and Ibata (2024) streams.

Accepts both official Gaia DR3 Archive schemas (`ra, dec, parallax, pmra, pmdec,
radial_velocity, phot_g_mean_mag, parallax_error, pmra_error, pmdec_error`) and
simplified stream tables such as `make_ICs/stars_6D_ibata2024.csv` (`RA, Dec,
Distance_kpc, pmRA, pmDec, Vlos, Stream_ID`), automatically converting `Distance_kpc`
back to linear parallax `1 / Distance_kpc` without 1/varpi singularities.

Supports 5D astrometric inference as well as automatic Best-4D fallback using
`GNN-jed/trained_models/study2_v2/best_4d_config.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gaia_v2 import (
    REF_SCALES_V2,
    encode_uncertainty_features_v2,
    gaia_dr3_astrometric_sigmas,
    mask_uncertainty_features,
    standardise_matrix_v2,
)
from gnn_models_v2 import CLASS_NAMES_3WAY, GAT_V2
from v2_build_graph_dataset import build_sky_knn_edges


COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ra": ("ra", "RA", "ra_deg", "RA_deg"),
    "dec": ("dec", "Dec", "DEC", "dec_deg", "Dec_deg"),
    "parallax": ("parallax", "plx", "plx_mas", "Parallax"),
    "distance_kpc": ("Distance_kpc", "distance_kpc", "dist_kpc", "d_kpc"),
    "pmra": ("pmra", "pmRA", "pm_ra"),
    "pmdec": ("pmdec", "pmDec", "pm_dec"),
    "vlos": ("radial_velocity", "Vlos", "vlos", "v_los", "rv"),
    "g_mag": ("phot_g_mean_mag", "G", "g_mag", "phot_g"),
    "plx_err": ("parallax_error", "plx_err", "sigma_plx"),
    "pmra_err": ("pmra_error", "pmRA_err", "sigma_pmra"),
    "pmdec_err": ("pmdec_error", "pmDec_err", "sigma_pmdec"),
}


def _resolve_col(header: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    hset = {h.strip(): h for h in header}
    for cand in aliases:
        if cand in hset:
            return hset[cand]
    return None


def _to_float_array(rows: Sequence[Dict[str, str]], col: Optional[str], default: float = np.nan) -> np.ndarray:
    if col is None:
        return np.full(len(rows), default, dtype=np.float64)
    vals = []
    for r in rows:
        raw = str(r.get(col, "")).strip()
        if not raw or raw.lower() in ("nan", "none", "null", "inf", "-inf"):
            vals.append(np.nan)
        else:
            try:
                vals.append(float(raw))
            except ValueError:
                vals.append(np.nan)
    return np.asarray(vals, dtype=np.float64)


def parse_gaia_csv(
    csv_path: str,
    group_col: Optional[str] = "Stream_ID",
) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray], np.ndarray]:
    """Read a Gaia/stream CSV into raw rows, standardized physical arrays, and group keys."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Input CSV '{csv_path}' contains no rows.")

    col_ra = _resolve_col(header, COLUMN_ALIASES["ra"])
    col_dec = _resolve_col(header, COLUMN_ALIASES["dec"])
    col_plx = _resolve_col(header, COLUMN_ALIASES["parallax"])
    col_dist = _resolve_col(header, COLUMN_ALIASES["distance_kpc"])
    col_pmra = _resolve_col(header, COLUMN_ALIASES["pmra"])
    col_pmdec = _resolve_col(header, COLUMN_ALIASES["pmdec"])
    col_vlos = _resolve_col(header, COLUMN_ALIASES["vlos"])
    col_gmag = _resolve_col(header, COLUMN_ALIASES["g_mag"])
    col_plx_err = _resolve_col(header, COLUMN_ALIASES["plx_err"])
    col_pmra_err = _resolve_col(header, COLUMN_ALIASES["pmra_err"])
    col_pmdec_err = _resolve_col(header, COLUMN_ALIASES["pmdec_err"])

    ra = _to_float_array(rows, col_ra)
    dec = _to_float_array(rows, col_dec)
    if col_plx is not None:
        plx = _to_float_array(rows, col_plx)
    elif col_dist is not None:
        dist_kpc = _to_float_array(rows, col_dist)
        plx = np.where(np.isfinite(dist_kpc) & (dist_kpc > 1e-3), 1.0 / np.clip(dist_kpc, 1e-3, None), np.nan)
    else:
        plx = np.full(len(rows), np.nan, dtype=np.float64)

    pmra = _to_float_array(rows, col_pmra)
    pmdec = _to_float_array(rows, col_pmdec)
    vlos = _to_float_array(rows, col_vlos, default=0.0)
    g_mag = _to_float_array(rows, col_gmag, default=18.5)
    g_mag = np.where(np.isfinite(g_mag), g_mag, 18.5)

    sig_default = gaia_dr3_astrometric_sigmas(g_mag, np.where(np.isfinite(dec), dec, 0.0))
    plx_err = _to_float_array(rows, col_plx_err)
    pmra_err = _to_float_array(rows, col_pmra_err)
    pmdec_err = _to_float_array(rows, col_pmdec_err)

    plx_err = np.where(np.isfinite(plx_err), plx_err, sig_default["sigma_plx_mas"])
    pmra_err = np.where(np.isfinite(pmra_err), pmra_err, sig_default["sigma_pmra_mas_yr"])
    pmdec_err = np.where(np.isfinite(pmdec_err), pmdec_err, sig_default["sigma_pmdec_mas_yr"])

    obs_raw = np.column_stack((ra, dec, plx, pmra, pmdec, np.where(np.isfinite(vlos), vlos, 0.0)))
    err_raw = np.column_stack((plx_err, pmra_err, pmdec_err))

    if group_col and group_col in header:
        groups = np.asarray([str(r[group_col]).strip() for r in rows])
    else:
        groups = np.asarray(["all"] * len(rows))

    arrays = {"obs_plx": obs_raw, "err_raw": err_raw}
    return rows, arrays, groups


def load_v2_checkpoint(
    weights_path: str,
    feat_indices: Sequence[int],
    hidden_channels: int = 64,
    heads: int = 4,
    out_channels: int = 3,
    edge_dim: int = 3,
    err_features: int = 4,
) -> GAT_V2:
    """Instantiate GAT_V2 and load trained weights."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    clean_state = {
        (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state.items()
    }
    if "head.4.bias" in clean_state:
        out_channels = int(clean_state["head.4.bias"].shape[0])
    if "lin_in.0.weight" in clean_state:
        hidden_channels = int(clean_state["lin_in.0.weight"].shape[0])

    model = GAT_V2(
        in_features=len(feat_indices),
        err_features=err_features,
        hidden_channels=hidden_channels,
        heads=heads,
        out_channels=out_channels,
        edge_dim=edge_dim,
        dropout=0.0,
    )
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model


def infer_group(
    obs_plx: np.ndarray,
    err_raw: np.ndarray,
    model: GAT_V2,
    feat_indices: Sequence[int],
    k_neighbors: int = 32,
) -> Optional[np.ndarray]:
    """Run GAT_V2 inference on one sky group. Returns (N, C) calibrated probabilities or None."""
    n_stars = len(obs_plx)
    if n_stars < 2:
        return None

    # Impute rare NaNs on non-sky cols ONLY if sky (cols 0,1) is finite
    if not np.all(np.isfinite(obs_plx[:, :2])):
        return None
    obs_clean = obs_plx.copy()
    for col in feat_indices:
        finite_mask = np.isfinite(obs_clean[:, col])
        if not np.any(finite_mask):
            return None
        if not np.all(finite_mask):
            obs_clean[~finite_mask, col] = float(np.median(obs_clean[finite_mask, col]))

    x_std, _ = standardise_matrix_v2(obs_clean, ref_scales=REF_SCALES_V2)
    edge_idx_np, edge_attr_np = build_sky_knn_edges(
        x_std[:, :2].astype(np.float64), k_neighbors=min(k_neighbors, n_stars - 1)
    )

    err_feats = encode_uncertainty_features_v2(
        np.where(np.isfinite(obs_clean[:, 2]), obs_clean[:, 2], 0.0),
        err_raw[:, 0],
        err_raw[:, 1],
        err_raw[:, 2],
    )
    err_feats = mask_uncertainty_features(err_feats, feat_indices)

    x_t = torch.from_numpy(np.ascontiguousarray(x_std[:, feat_indices], dtype=np.float32))
    err_t = torch.from_numpy(np.ascontiguousarray(err_feats, dtype=np.float32))
    edge_idx_t = torch.from_numpy(edge_idx_np)
    edge_attr_t = torch.from_numpy(edge_attr_np)

    with torch.no_grad():
        logits = model(x_t, edge_idx_t, edge_attr=edge_attr_t, err_x=err_t)
        if logits.shape[1] == 1:
            p_b = torch.sigmoid(logits.squeeze(-1)).numpy()
            probs = np.column_stack((1.0 - p_b, np.zeros_like(p_b), p_b))
        else:
            probs = torch.softmax(logits, dim=-1).numpy()

    return probs


def run_inference(
    input_csv: str,
    out_csv: str,
    out_summary: str,
    model_5d_path: str = "",
    model_4d_path: str = "",
    config_4d_path: str = "",
    group_col: str = "Stream_ID",
    group_filter: str = "",
    mode: str = "auto",
    k_neighbors: int = 32,
) -> Dict[str, Any]:
    """Execute Study 2 V2 catalog inference and write enriched CSV + JSON summary."""
    rows, arrays, groups = parse_gaia_csv(input_csv, group_col=group_col)
    obs_plx = arrays["obs_plx"]
    err_raw = arrays["err_raw"]

    feat_4d = [0, 1, 3, 4]
    tag_4d = "4obs_default"
    if config_4d_path and os.path.isfile(config_4d_path):
        with open(config_4d_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        feat_4d = [int(i) for i in cfg["feature_indices"]]
        tag_4d = str(cfg.get("best_tag", tag_4d))

    model_5d = (
        load_v2_checkpoint(model_5d_path, [0, 1, 2, 3, 4])
        if (model_5d_path and os.path.isfile(model_5d_path))
        else None
    )
    model_4d = (
        load_v2_checkpoint(model_4d_path, feat_4d)
        if (model_4d_path and os.path.isfile(model_4d_path))
        else None
    )

    if model_5d is None and model_4d is None:
        raise RuntimeError("Neither a valid --model_5d nor --model_4d checkpoint was provided.")

    n_rows = len(rows)
    probs_all = np.full((n_rows, 3), np.nan, dtype=np.float32)
    used_model_tag = ["none"] * n_rows

    unique_groups = sorted(set(groups.tolist()), key=lambda s: (int(s) if s.isdigit() else 999999, s))
    if group_filter:
        unique_groups = [g for g in unique_groups if g == group_filter]

    group_summaries: List[Dict[str, Any]] = []
    for grp in unique_groups:
        idx = np.flatnonzero(groups == grp)
        g_obs = obs_plx[idx]
        g_err = err_raw[idx]

        has_5d = bool(np.all(np.isfinite(g_obs[:, :5]).mean(axis=0) > 0.5))
        has_4d = bool(np.all(np.isfinite(g_obs[:, feat_4d]).mean(axis=0) > 0.5))

        chosen_model = None
        chosen_feats: Sequence[int] = [0, 1, 2, 3, 4]
        chosen_tag = "none"

        if mode == "5d" and model_5d is not None and has_5d:
            chosen_model, chosen_feats, chosen_tag = model_5d, [0, 1, 2, 3, 4], "5obs"
        elif mode == "4d" and model_4d is not None and has_4d:
            chosen_model, chosen_feats, chosen_tag = model_4d, feat_4d, tag_4d
        elif mode == "auto":
            if model_5d is not None and has_5d:
                chosen_model, chosen_feats, chosen_tag = model_5d, [0, 1, 2, 3, 4], "5obs"
            elif model_4d is not None and has_4d:
                chosen_model, chosen_feats, chosen_tag = model_4d, feat_4d, tag_4d

        if chosen_model is None:
            continue

        g_probs = infer_group(g_obs, g_err, chosen_model, chosen_feats, k_neighbors=k_neighbors)
        if g_probs is None:
            continue

        probs_all[idx] = g_probs
        for i_row in idx:
            used_model_tag[i_row] = chosen_tag

        pred_cls = np.argmax(g_probs, axis=-1)
        n_g = len(idx)
        group_summaries.append(
            {
                "group": grp,
                "n_stars": int(n_g),
                "model_used": chosen_tag,
                "n_unbound_field": int(np.sum(pred_cls == 0)),
                "n_potential_escaper": int(np.sum(pred_cls == 1)),
                "n_bound_core": int(np.sum(pred_cls == 2)),
                "bound_fraction": float(np.mean(pred_cls == 2)),
                "mean_p_bound": float(np.mean(g_probs[:, 2])),
                "mean_p_pe": float(np.mean(g_probs[:, 1])),
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    extra_cols = [
        "V2_Model_Used",
        "P_Unbound_Field",
        "P_Potential_Escaper",
        "P_Bound_Core",
        "Pred_Class",
        "Pred_Label",
    ]
    base_cols = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + extra_cols)
        writer.writeheader()
        for r_i, row in enumerate(rows):
            out_row = dict(row)
            p = probs_all[r_i]
            out_row["V2_Model_Used"] = used_model_tag[r_i]
            if np.all(np.isfinite(p)):
                cls_i = int(np.argmax(p))
                out_row["P_Unbound_Field"] = f"{p[0]:.6f}"
                out_row["P_Potential_Escaper"] = f"{p[1]:.6f}"
                out_row["P_Bound_Core"] = f"{p[2]:.6f}"
                out_row["Pred_Class"] = str(cls_i)
                out_row["Pred_Label"] = CLASS_NAMES_3WAY[cls_i]
            else:
                out_row["P_Unbound_Field"] = ""
                out_row["P_Potential_Escaper"] = ""
                out_row["P_Bound_Core"] = ""
                out_row["Pred_Class"] = ""
                out_row["Pred_Label"] = "Skipped"
            writer.writerow(out_row)

    summary = {
        "input_csv": input_csv,
        "out_csv": out_csv,
        "total_rows": n_rows,
        "groups_evaluated": len(group_summaries),
        "groups": group_summaries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_summary)), exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply trained Study 2 V2 GAT_V2 models to Gaia DR3 star catalogs."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_summary", type=str, required=True)
    parser.add_argument("--model_5d", type=str, default="")
    parser.add_argument("--model_4d", type=str, default="")
    parser.add_argument("--config_4d", type=str, default="")
    parser.add_argument("--group_col", type=str, default="Stream_ID")
    parser.add_argument("--group_filter", type=str, default="")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "5d", "4d"])
    parser.add_argument("--k_neighbors", type=int, default=32)
    args = parser.parse_args()

    summary = run_inference(
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        model_5d_path=args.model_5d,
        model_4d_path=args.model_4d,
        config_4d_path=args.config_4d,
        group_col=args.group_col,
        group_filter=args.group_filter,
        mode=args.mode,
        k_neighbors=args.k_neighbors,
    )
    print(
        f"Gaia DR3 V2 inference complete: {summary['total_rows']} stars across "
        f"{summary['groups_evaluated']} groups -> '{args.out_csv}'."
    )


if __name__ == "__main__":
    main()
