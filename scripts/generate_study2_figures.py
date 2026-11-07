"""Generate publication-quality PNG figures and interactive HTML visualizations for Study 2.

Reproduces all diagnostic plots from the MA2 report and `README.md` on the Study 2 dataset
and models (Stage 1 Clean pre-training, Stage 2 Gaia DR3 Noisy fine-tuning, and 4D ablations):
1. Progenitor seeds and orbital pericenter distribution (`seeds_positions_unbiased.{png,html}`)
2. Galactocentric radius R(t) colored by bound fraction f_bound(t) across 6 R_{p1} panels (`R_vs_t_6panels_rp1.{png,html}`)
3. Training & Validation Loss/Accuracy curves (`live_plots_GNN_{tag}.{png,html}`)
4. Singular Value Decomposition (SVD) weight spectrum (`spectral_analysis_GNN_{tag}.{png,html}`)
5. Spatial classification maps TP/TN/FP/FN (`prediction_{model}_ID{sid}_output100_{tag}.{png,html}`)
6. Classification error vs. first perigalacticon R_{p1} (`error_percentage_by_perigalacticon_{model}_{tag}.{png,html}`)
7. Unified interactive HTML dashboard (`study2_interactive_dashboard_{tag}.html`)
"""

import argparse
import csv
import json
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn_models import GCN as GCNModel, GAT as GATModel


def _write_plotly_html(out_html, title, traces, layout):
    """Write a standalone interactive Plotly HTML file."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 20px;
      background-color: #0f172a;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    h2 {{
      margin-top: 0;
      font-weight: 600;
      color: #e2e8f0;
    }}
    #plot {{
      width: 100%;
      height: 82vh;
      background: #1e293b;
      border-radius: 10px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div id="plot"></div>
  <script>
    const traces = {json.dumps(traces)};
    const layout = Object.assign({{
      paper_bgcolor: "#1e293b",
      plot_bgcolor: "#0f172a",
      font: {{ color: "#e2e8f0", size: 13 }},
      hovermode: "closest"
    }}, {json.dumps(layout)});
    Plotly.newPlot("plot", traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)


def parse_training_log(log_path):
    """Parse epoch metrics from training_metrics_*.log."""
    epochs, tr_loss, tr_acc, val_loss, val_acc = [], [], [], [], []
    if not log_path or not os.path.exists(log_path):
        return epochs, tr_loss, tr_acc, val_loss, val_acc

    pat = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%"
    )
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_acc.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))
    return epochs, tr_loss, tr_acc, val_loss, val_acc


def plot_training_curves(gcn_log, gat_log, output_dir, tag):
    """Generate live_plots_GNN_{tag}.png and .html."""
    ep1, tr_l1, tr_a1, val_l1, val_a1 = parse_training_log(gcn_log)
    ep2, tr_l2, tr_a2, val_l2, val_a2 = parse_training_log(gat_log)
    if not ep1 and not ep2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    if ep1:
        ax1.plot(ep1, tr_l1, "--", color="#3b82f6", alpha=0.8, label="GCN Train Loss")
        ax1.plot(ep1, val_l1, "-", color="#1d4ed8", lw=2.2, label="GCN Val Loss")
        ax2.plot(ep1, tr_a1, "--", color="#3b82f6", alpha=0.8, label="GCN Train Acc")
        ax2.plot(ep1, val_a1, "-", color="#1d4ed8", lw=2.2, label="GCN Val Acc")
    if ep2:
        ax1.plot(ep2, tr_l2, "--", color="#f97316", alpha=0.8, label="GAT Train Loss")
        ax1.plot(ep2, val_l2, "-", color="#ea580c", lw=2.2, label="GAT Val Loss")
        ax2.plot(ep2, tr_a2, "--", color="#f97316", alpha=0.8, label="GAT Train Acc")
        ax2.plot(ep2, val_a2, "-", color="#ea580c", lw=2.2, label="GAT Val Acc")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.set_title(f"Training & Validation Loss ({tag})")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_title(f"Training & Validation Accuracy ({tag})")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"live_plots_GNN_{tag}.png")
    plt.savefig(png_path)
    plt.close(fig)

    traces = []
    if ep1:
        traces.extend([
            {"x": ep1, "y": tr_l1, "mode": "lines", "name": "GCN Train Loss", "line": {"dash": "dash", "color": "#60a5fa"}, "xaxis": "x1", "yaxis": "y1"},
            {"x": ep1, "y": val_l1, "mode": "lines", "name": "GCN Val Loss", "line": {"width": 3, "color": "#2563eb"}, "xaxis": "x1", "yaxis": "y1"},
            {"x": ep1, "y": tr_a1, "mode": "lines", "name": "GCN Train Acc (%)", "line": {"dash": "dash", "color": "#60a5fa"}, "xaxis": "x2", "yaxis": "y2"},
            {"x": ep1, "y": val_a1, "mode": "lines", "name": "GCN Val Acc (%)", "line": {"width": 3, "color": "#2563eb"}, "xaxis": "x2", "yaxis": "y2"},
        ])
    if ep2:
        traces.extend([
            {"x": ep2, "y": tr_l2, "mode": "lines", "name": "GAT Train Loss", "line": {"dash": "dash", "color": "#fb923c"}, "xaxis": "x1", "yaxis": "y1"},
            {"x": ep2, "y": val_l2, "mode": "lines", "name": "GAT Val Loss", "line": {"width": 3, "color": "#ea580c"}, "xaxis": "x1", "yaxis": "y1"},
            {"x": ep2, "y": tr_a2, "mode": "lines", "name": "GAT Train Acc (%)", "line": {"dash": "dash", "color": "#fb923c"}, "xaxis": "x2", "yaxis": "y2"},
            {"x": ep2, "y": val_a2, "mode": "lines", "name": "GAT Val Acc (%)", "line": {"width": 3, "color": "#ea580c"}, "xaxis": "x2", "yaxis": "y2"},
        ])
    layout = {
        "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
        "xaxis1": {"title": "Epoch"},
        "yaxis1": {"title": "BCE Loss"},
        "xaxis2": {"title": "Epoch"},
        "yaxis2": {"title": "Accuracy [%]"},
    }
    _write_plotly_html(
        os.path.join(output_dir, f"live_plots_GNN_{tag}.html"),
        f"GCN vs. GAT Training & Validation Dynamics ({tag})",
        traces,
        layout,
    )


def plot_spectral_analysis(gcn_weights_path, gat_weights_path, output_dir, tag, in_channels=5):
    """Perform SVD on the second layer weight matrices of GCN and GAT."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=180)
    traces = []

    if gcn_weights_path and os.path.exists(gcn_weights_path):
        m_gcn = GCNModel(in_channels=in_channels)
        m_gcn.load_state_dict(torch.load(gcn_weights_path, map_location="cpu", weights_only=True))
        w_gcn = m_gcn.conv2.lin.weight.detach().cpu().numpy()
        s_gcn = np.linalg.svd(w_gcn, compute_uv=False)
        s_gcn_norm = s_gcn / max(1e-12, s_gcn[0])
        idx_gcn = list(range(1, len(s_gcn_norm) + 1))
        ax.semilogy(idx_gcn, s_gcn_norm, "o-", color="#2563eb", lw=2, ms=4, label="GCN (conv2: 32x64)")
        traces.append({
            "x": idx_gcn,
            "y": s_gcn_norm.tolist(),
            "mode": "lines+markers",
            "name": "GCN (conv2: 32x64)",
            "line": {"color": "#3b82f6", "width": 2.5},
        })

    if gat_weights_path and os.path.exists(gat_weights_path):
        m_gat = GATModel(in_channels=in_channels)
        m_gat.load_state_dict(torch.load(gat_weights_path, map_location="cpu", weights_only=True))
        w_gat_tensor = (
            m_gat.conv2.lin.weight
            if hasattr(m_gat.conv2, "lin") and m_gat.conv2.lin is not None
            else m_gat.conv2.lin_src.weight
        )
        w_gat = w_gat_tensor.detach().cpu().numpy()
        s_gat = np.linalg.svd(w_gat, compute_uv=False)
        s_gat_norm = s_gat / max(1e-12, s_gat[0])
        idx_gat = list(range(1, len(s_gat_norm) + 1))
        ax.semilogy(idx_gat, s_gat_norm, "s-", color="#ea580c", lw=2, ms=3.5, label="GAT (conv2: 128x256)")
        traces.append({
            "x": idx_gat,
            "y": s_gat_norm.tolist(),
            "mode": "lines+markers",
            "name": "GAT (conv2: 128x256)",
            "line": {"color": "#f97316", "width": 2.5},
        })

    if not traces:
        plt.close(fig)
        return

    ax.set_xlabel("Singular Value Index $i$")
    ax.set_ylabel("Normalized Singular Value $\\sigma_i / \\sigma_1$")
    ax.set_title(f"Singular Value Spectrum of Intermediate Layer Weights ({tag})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spectral_analysis_GNN_{tag}.png"))
    plt.close(fig)

    _write_plotly_html(
        os.path.join(output_dir, f"spectral_analysis_GNN_{tag}.html"),
        f"Singular Value Spectrum (SVD) of GNN Weight Matrices ({tag})",
        traces,
        {"xaxis": {"title": "Singular Value Index i"}, "yaxis": {"title": "Normalized Singular Value (sigma_i / sigma_1)", "type": "log"}},
    )


def plot_seeds_and_orbits_summary(summary_csv, output_dir):
    """Plot initial galactocentric distance vs perigalacticon R_p and 3D progenitor seeds (.png & .html)."""
    if not os.path.exists(summary_csv):
        return

    sel_r_init, sel_rp, sel_ecc, sel_lbl = [], [], [], []
    rej_r_init, rej_rp, rej_ecc, rej_lbl = [], [], [], []
    sel_x, sel_y, sel_z = [], [], []

    with open(summary_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["Stream_ID"])
            r_init = float(row["r_init_2Gyr_kpc"])
            rp = float(row["R_peri_kpc"])
            ecc = float(row["eccentricity"])
            lbl = f"Stream {sid} (R_p={rp:.2f} kpc, r_init={r_init:.1f} kpc, e={ecc:.2f})"
            if str(row["selected"]).lower() == "true":
                sel_r_init.append(r_init)
                sel_rp.append(rp)
                sel_ecc.append(ecc)
                sel_lbl.append(lbl)
                sel_x.append(float(row["x_init_kpc"]))
                sel_y.append(float(row["y_init_kpc"]))
                sel_z.append(float(row["z_init_kpc"]))
            else:
                rej_r_init.append(r_init)
                rej_rp.append(rp)
                rej_ecc.append(ecc)
                rej_lbl.append(lbl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=180)
    ax1.scatter(rej_r_init, rej_rp, c="#ef4444", marker="x", s=45, alpha=0.8, label=f"Rejected (R_p < 5 kpc or out of box: {len(rej_rp)})")
    sc = ax1.scatter(sel_r_init, sel_rp, c=sel_ecc, cmap="viridis", s=55, edgecolors="k", linewidth=0.5, label=f"Selected Study 2 Progenitors ({len(sel_rp)})")
    ax1.axhline(5.0, color="#ef4444", linestyle="--", lw=1.5, label="Perigalacticon Threshold $R_p = 5$ kpc")
    ax1.axvline(100.0, color="#64748b", linestyle=":", lw=1.5, label="Study 1 Cutoff $r(-2\\,\\mathrm{Gyr}) = 100$ kpc")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Initial Galactocentric Distance $r(-2\\,\\mathrm{Gyr})$ [kpc]")
    ax1.set_ylabel("Perigalacticon $R_p$ [kpc]")
    ax1.set_title("Unbiased Progenitor Orbit Selection (Ibata et al. 2024)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8.5)
    fig.colorbar(sc, ax=ax1, label="Orbital Eccentricity $e$")

    sc2 = ax2.scatter(sel_y, sel_z, c=sel_rp, cmap="plasma", s=60, edgecolors="k", linewidth=0.5)
    ax2.set_xlabel("Initial Galactocentric $Y(-2\\,\\mathrm{Gyr})$ [kpc]")
    ax2.set_ylabel("Initial Galactocentric $Z(-2\\,\\mathrm{Gyr})$ [kpc]")
    ax2.set_title("Selected Progenitor Seeds at $t = -2$ Gyr ($Y$-$Z$ Projection)")
    ax2.grid(True, alpha=0.25)
    fig.colorbar(sc2, ax=ax2, label="Perigalacticon $R_p$ [kpc]")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seeds_positions_unbiased.png"))
    plt.close(fig)

    _write_plotly_html(
        os.path.join(output_dir, "seeds_positions_unbiased.html"),
        "Study 2 Unbiased Progenitor Orbit Selection (50 Selected vs. 34 Rejected Streams)",
        [
            {
                "x": rej_r_init,
                "y": rej_rp,
                "mode": "markers",
                "name": f"Rejected ({len(rej_rp)})",
                "text": rej_lbl,
                "hoverinfo": "text",
                "marker": {"color": "#ef4444", "symbol": "x", "size": 9},
            },
            {
                "x": sel_r_init,
                "y": sel_rp,
                "mode": "markers",
                "name": f"Selected R_p >= 5 kpc ({len(sel_rp)})",
                "text": sel_lbl,
                "hoverinfo": "text",
                "marker": {
                    "color": sel_ecc,
                    "colorscale": "Viridis",
                    "size": 11,
                    "colorbar": {"title": "Eccentricity e"},
                },
            },
        ],
        {
            "xaxis": {"title": "Initial Galactocentric Distance r(-2 Gyr) [kpc]", "type": "log"},
            "yaxis": {"title": "Perigalacticon R_p [kpc]", "type": "log"},
        },
    )


def plot_r_vs_t_panels(summary_csv, orbits_dir, labels_dir, output_dir):
    """Plot R(t) colored by bound fraction across 6 panels ordered by first pericenter R_{p1}."""
    if not os.path.exists(summary_csv) or not os.path.exists(labels_dir):
        return

    streams = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("selected", "")).lower() == "true":
                sid = int(row["Stream_ID"])
                rp1 = float(row.get("R_p1_kpc", row["R_peri_kpc"]))
                streams.append((sid, rp1, float(row["eccentricity"])))

    streams.sort(key=lambda x: x[1])
    if not streams:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=180, sharex=True)
    axes_flat = axes.flatten()
    chunks = np.array_split(np.arange(len(streams)), 6)
    t_grid = np.linspace(0.0, 2.0, 101)
    norm = plt.Normalize(0.0, 1.0)
    html_traces = []

    for panel_idx, idx_chunk in enumerate(chunks):
        ax = axes_flat[panel_idx]
        if len(idx_chunk) == 0:
            continue
        rp_min = streams[idx_chunk[0]][1]
        rp_max = streams[idx_chunk[-1]][1]

        for s_idx in idx_chunk:
            sid, rp1, ecc = streams[s_idx]
            orbit_file = os.path.join(orbits_dir, f"orbit_{sid}.txt")
            ic_label_dir = os.path.join(labels_dir, f"IC_{sid}")
            if not (os.path.exists(orbit_file) and os.path.exists(ic_label_dir)):
                continue

            orb = np.loadtxt(orbit_file, skiprows=1)
            r_orb = np.sqrt(orb[:, 0] ** 2 + orb[:, 1] ** 2 + orb[:, 2] ** 2)[::-1]
            r_snap = np.interp(t_grid, np.linspace(0.0, 2.0, len(r_orb)), r_orb)

            f_bound = []
            for snap in range(101):
                lbl_path = os.path.join(ic_label_dir, f"B_L_snapshot_{snap:04d}.npy")
                if os.path.exists(lbl_path):
                    arr = np.load(lbl_path)
                    f_bound.append(float(np.mean(arr)))
                else:
                    f_bound.append(np.nan)
            f_bound = np.asarray(f_bound, dtype=float)
            valid = ~np.isnan(f_bound)
            if np.sum(valid) < 2:
                continue

            pts = np.array([t_grid[valid], r_snap[valid]]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="magma", norm=norm, linewidth=2.0, alpha=0.9)
            lc.set_array(f_bound[valid][:-1])
            ax.add_collection(lc)

            html_traces.append({
                "x": t_grid[valid].tolist(),
                "y": r_snap[valid].tolist(),
                "mode": "lines+markers",
                "name": f"IC_{sid} (R_p1={rp1:.1f} kpc, e={ecc:.2f})",
                "marker": {
                    "color": f_bound[valid].tolist(),
                    "colorscale": "Magma",
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "size": 5,
                },
                "line": {"width": 2},
                "text": [f"IC_{sid} | t={t:.2f} Gyr | R={r:.1f} kpc | f_bound={fb*100:.1f}%" for t, r, fb in zip(t_grid[valid], r_snap[valid], f_bound[valid])],
                "hoverinfo": "text",
            })

        ax.autoscale()
        ax.set_xlim(0.0, 2.0)
        ax.set_title(f"Panel {panel_idx+1}: $R_{{p1}} \\in [{rp_min:.1f}, {rp_max:.1f}]$ kpc")
        ax.grid(True, alpha=0.25)
        if panel_idx >= 3:
            ax.set_xlabel("Simulation Time $t$ [Gyr]")
        if panel_idx % 3 == 0:
            ax.set_ylabel("Galactocentric Distance $R(t)$ [kpc]")

    sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Bound Star Fraction $f_{\\mathrm{bound}}(t)$", pad=0.02)
    plt.savefig(os.path.join(output_dir, "R_vs_t_6panels_rp1.png"), bbox_inches="tight")
    plt.close(fig)

    if html_traces:
        _write_plotly_html(
            os.path.join(output_dir, "R_vs_t_6panels_rp1.html"),
            "Galactocentric Distance R(t) Colored by Bound Star Fraction f_bound(t)",
            html_traces,
            {"xaxis": {"title": "Simulation Time t [Gyr]"}, "yaxis": {"title": "Galactocentric Distance R(t) [kpc]"}},
        )


def evaluate_per_stream_and_plot(
    model_path,
    model_type,
    graph_dir,
    sim_dir,
    summary_csv,
    output_dir,
    tag,
    in_channels=5,
    feature_indices=None,
    sample_ids=(1, 11, 53, 70, 74, 81),
):
    """Evaluate trained model at snapshot 100 across all streams and plot spatial maps & error vs R_{p1}."""
    if not model_path or not os.path.exists(model_path) or not os.path.exists(graph_dir):
        return

    model = (
        GATModel(in_channels=in_channels)
        if model_type.upper() == "GAT"
        else GCNModel(in_channels=in_channels)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    orbit_info = {}
    if os.path.exists(summary_csv):
        with open(summary_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = int(row["Stream_ID"])
                orbit_info[sid] = {
                    "R_p1": float(row.get("R_p1_kpc", row["R_peri_kpc"])),
                    "R_apo": float(row["R_apo_kpc"]),
                    "ecc": float(row["eccentricity"]),
                }

    stream_errors = []
    ic_dirs = sorted(
        [d for d in os.listdir(graph_dir) if d.startswith("IC_") and os.path.isdir(os.path.join(graph_dir, d))],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
    )

    for ic_name in ic_dirs:
        sid = int(ic_name.split("_")[1])
        g_file = os.path.join(graph_dir, ic_name, "graph_data_output_100_rotation_0.pt")
        l_file = os.path.join(graph_dir, ic_name, "label_output_100_rotation_0.pt")
        if not (os.path.exists(g_file) and os.path.exists(l_file)):
            continue

        data = torch.load(g_file, weights_only=False)
        if feature_indices is not None and data.x.shape[1] > len(feature_indices):
            data.x = data.x[:, feature_indices]
        y_true = torch.load(l_file, weights_only=False).float().numpy()

        with torch.no_grad():
            logits = model(data).view(-1)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(np.float32)

        err_pct = float(np.mean(preds != y_true) * 100.0)
        f_bound_true = float(np.mean(y_true) * 100.0)
        rp1 = orbit_info.get(sid, {}).get("R_p1", float(sid))
        ecc = orbit_info.get(sid, {}).get("ecc", 0.0)
        stream_errors.append((sid, rp1, ecc, err_pct, f_bound_true))

        if sid in sample_ids:
            snap_hdf5 = os.path.join(sim_dir, ic_name, "snapshot_0100.hdf5")
            if os.path.exists(snap_hdf5):
                try:
                    from plot_swift_output import load_snapshot
                    pos_raw, _, _, _ = load_snapshot(snap_hdf5, shift=500.0)
                    coord_x, coord_y = pos_raw[:, 1], pos_raw[:, 2]
                    xlab, ylab = "Galactocentric Y [kpc]", "Galactocentric Z [kpc]"
                except Exception:
                    coord_x, coord_y = data.x[:, 0].numpy(), data.x[:, 1].numpy()
                    xlab, ylab = "Standardized Feature 0 (RA*)", "Standardized Feature 1 (Dec)"
            else:
                coord_x, coord_y = data.x[:, 0].numpy(), data.x[:, 1].numpy()
                xlab, ylab = "Standardized Feature 0 (RA*)", "Standardized Feature 1 (Dec)"

            tn = (y_true == 0) & (preds == 0)
            tp = (y_true == 1) & (preds == 1)
            fn = (y_true == 1) & (preds == 0)
            fp = (y_true == 0) & (preds == 1)

            fig, ax = plt.subplots(figsize=(8.5, 7), dpi=180)
            ax.scatter(coord_x[tn], coord_y[tn], s=2, c="#94a3b8", alpha=0.45, label=f"True Unbound (TN: {np.sum(tn)})")
            ax.scatter(coord_x[tp], coord_y[tp], s=4, c="#2563eb", alpha=0.8, label=f"True Bound (TP: {np.sum(tp)})")
            ax.scatter(coord_x[fp], coord_y[fp], s=8, c="#f97316", alpha=0.9, label=f"False Bound (FP: {np.sum(fp)})")
            ax.scatter(coord_x[fn], coord_y[fn], s=8, c="#dc2626", alpha=0.9, label=f"Missed Bound (FN: {np.sum(fn)})")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(f"{model_type} Prediction on IC_{sid} at t = 2.0 Gyr ({tag} | Error: {err_pct:.2f}%)")
            ax.legend(loc="upper right", markerscale=2.0)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            base_pred = f"prediction_{model_type}_ID{sid}_output100_{tag}"
            plt.savefig(os.path.join(output_dir, f"{base_pred}.png"))
            plt.close(fig)

            pred_traces = []
            for mask, name, col, sz in [
                (tn, "True Unbound (TN)", "#64748b", 3),
                (tp, "True Bound (TP)", "#3b82f6", 5),
                (fp, "False Bound (FP)", "#f97316", 7),
                (fn, "Missed Bound (FN)", "#ef4444", 7),
            ]:
                if np.any(mask):
                    pred_traces.append({
                        "x": coord_x[mask].tolist(),
                        "y": coord_y[mask].tolist(),
                        "mode": "markers",
                        "type": "scattergl",
                        "name": f"{name} ({int(np.sum(mask))})",
                        "marker": {"color": col, "size": sz, "opacity": 0.8},
                    })
            _write_plotly_html(
                os.path.join(output_dir, f"{base_pred}.html"),
                f"{model_type} Spatial Classification on IC_{sid} at t = 2.0 Gyr ({tag} | Error: {err_pct:.2f}%)",
                pred_traces,
                {"xaxis": {"title": xlab}, "yaxis": {"title": ylab, "scaleanchor": "x", "scaleratio": 1}},
            )

    if not stream_errors:
        return

    stream_errors.sort(key=lambda x: x[1])
    labels_x = [f"IC_{x[0]}" for x in stream_errors]
    rp1_vals = [x[1] for x in stream_errors]
    ecc_vals = [x[2] for x in stream_errors]
    err_vals = [x[3] for x in stream_errors]
    fb_vals = [x[4] for x in stream_errors]
    mean_err = float(np.mean(err_vals))

    fig, ax = plt.subplots(figsize=(15, 5.5), dpi=180)
    bars = ax.bar(range(len(stream_errors)), err_vals, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
    ax.axhline(mean_err, color="#dc2626", linestyle="--", lw=2, label=f"Mean Error: {mean_err:.2f}% (Acc: {100-mean_err:.2f}%)")
    ax.set_xticks(range(len(stream_errors)))
    ax.set_xticklabels(labels_x, rotation=75, fontsize=8)
    ax.set_xlabel("Simulated Stellar Streams Ordered by Increasing First Perigalacticon $R_{p1}$ [kpc]")
    ax.set_ylabel("Classification Error at $t = 2$ Gyr [%]")
    ax.set_title(f"{model_type} Node Classification Error by Perigalacticon $R_{{p1}}$ ({tag})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    err_base = f"error_percentage_by_perigalacticon_{model_type}_{tag}"
    plt.savefig(os.path.join(output_dir, f"{err_base}.png"))
    plt.close(fig)

    hover_text = [
        f"{lbl} | R_p1={rp:.2f} kpc | e={ec:.2f} | Error={er:.2f}% | Bound={fb:.1f}%"
        for lbl, rp, ec, er, fb in zip(labels_x, rp1_vals, ecc_vals, err_vals, fb_vals)
    ]
    _write_plotly_html(
        os.path.join(output_dir, f"{err_base}.html"),
        f"{model_type} Classification Error at t = 2.0 Gyr Ordered by R_p1 ({tag} | Mean Error: {mean_err:.2f}%)",
        [
            {
                "x": labels_x,
                "y": err_vals,
                "type": "bar",
                "name": "Classification Error (%)",
                "text": hover_text,
                "hoverinfo": "text",
                "marker": {"color": rp1_vals, "colorscale": "Viridis", "colorbar": {"title": "R_p1 [kpc]"}},
            },
            {
                "x": labels_x,
                "y": [mean_err] * len(labels_x),
                "mode": "lines",
                "name": f"Mean Error ({mean_err:.2f}%)",
                "line": {"color": "#ef4444", "dash": "dash", "width": 2.5},
            },
        ],
        {"xaxis": {"title": "Streams Ordered by First Perigalacticon R_p1"}, "yaxis": {"title": "Error Percentage [%]"}},
    )


def build_master_dashboard(output_dir, tag):
    """Create a single index HTML page linking and displaying all generated HTML/PNG figures."""
    html_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".html") and not f.startswith("study2_interactive_dashboard")])
    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])

    links_html = "\n".join(
        f'<li><a href="{f}" target="_blank">{f}</a></li>' for f in html_files
    )
    cards_html = "\n".join(
        f'<div class="card"><h3>{f}</h3><img src="{f}" alt="{f}" /></div>' for f in png_files
    )

    dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Study 2 Interactive Figure Dashboard ({tag})</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 28px;
    }}
    h1, h2 {{ color: #e2e8f0; }}
    a {{ color: #38bdf8; text-decoration: none; font-weight: 500; }}
    a:hover {{ text-decoration: underline; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }}
    .card {{
      background: #1e293b;
      padding: 16px;
      border-radius: 10px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    }}
    .card img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>Study 2 Scientific Figures Dashboard ({tag})</h1>
  <h2>Interactive HTML Plots (Click to open full-screen zoomable Plotly view)</h2>
  <ul>
    {links_html}
  </ul>
  <h2>Static Publication PNG Figures</h2>
  <div class="grid">
    {cards_html}
  </div>
</body>
</html>
"""
    with open(os.path.join(output_dir, f"study2_interactive_dashboard_{tag}.html"), "w", encoding="utf-8") as f:
        f.write(dashboard)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PNG and interactive HTML figures for Study 2."
    )
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2"), help="Output directory for figures.")
    parser.add_argument("--tag", type=str, default="clean_5obs", help="Tag for figure filenames (e.g. 'clean_5obs', 'noisy_5obs', 'ablation_no_parallax').")
    parser.add_argument("--gcn_log", type=str, default="", help="Path to GCN training log.")
    parser.add_argument("--gat_log", type=str, default="", help="Path to GAT training log.")
    parser.add_argument("--gcn_weights", type=str, default="", help="Path to trained GCN .pth weights.")
    parser.add_argument("--gat_weights", type=str, default="", help="Path to trained GAT .pth weights.")
    parser.add_argument("--graph_dir", type=str, default="/scratch/cozma/graph_data_clean", help="Path to graph_data directory.")
    parser.add_argument("--sim_dir", type=str, default="/home/cozma/swift_sims/outputs_unbiased", help="Path to SWIFT simulation outputs.")
    parser.add_argument("--labels_dir", type=str, default=os.path.join(ROOT_DIR, "GNN-jed", "labels_clean"), help="Path to bound labels directory.")
    parser.add_argument("--summary_csv", type=str, default=os.path.join(ROOT_DIR, "make_ICs", "unbiased_orbits_summary.csv"), help="Path to unbiased_orbits_summary.csv.")
    parser.add_argument("--orbits_dir", type=str, default=os.path.join(ROOT_DIR, "make_ICs", "orbits_unbiased"), help="Path to orbits_unbiased directory.")
    parser.add_argument("--in_channels", type=int, default=5, help="Number of input features.")
    parser.add_argument("--feature_indices", type=str, default="", help="Optional comma-separated feature column indices.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    feat_idx = (
        [int(x.strip()) for x in args.feature_indices.split(",") if x.strip() != ""]
        if args.feature_indices
        else None
    )

    plot_seeds_and_orbits_summary(args.summary_csv, args.output_dir)
    plot_r_vs_t_panels(args.summary_csv, args.orbits_dir, args.labels_dir, args.output_dir)
    plot_training_curves(args.gcn_log, args.gat_log, args.output_dir, args.tag)
    plot_spectral_analysis(args.gcn_weights, args.gat_weights, args.output_dir, args.tag, in_channels=args.in_channels)

    if args.gat_weights and os.path.exists(args.gat_weights):
        evaluate_per_stream_and_plot(
            args.gat_weights, "GAT", args.graph_dir, args.sim_dir, args.summary_csv, args.output_dir, args.tag,
            in_channels=args.in_channels, feature_indices=feat_idx
        )
    if args.gcn_weights and os.path.exists(args.gcn_weights):
        evaluate_per_stream_and_plot(
            args.gcn_weights, "GCN", args.graph_dir, args.sim_dir, args.summary_csv, args.output_dir, args.tag,
            in_channels=args.in_channels, feature_indices=feat_idx
        )

    build_master_dashboard(args.output_dir, args.tag)
    print(f"All PNG and interactive HTML figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
