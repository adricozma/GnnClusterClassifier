"""Real-time training curve plotter for Study 2 GNN experiments.

Generates publication-grade live training monitoring figures (`live_training_curves.png`
and `GNN-jed/trained_models/study2/latest_live_training.png`) at the end of every training
epoch so learning dynamics (BCE loss, node classification accuracy, best validation
checkpoints, and learning rate schedules) can be inspected live during SLURM execution.
"""

import glob
import os
import re

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EPOCH_LINE_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.eE+-]+)(?:.*?(\(\*\)))?"
)
CORE_FNR_RE = re.compile(r"(?:Core\s*FNR|CoreFNR|FNR)\s*:\s*([0-9.]+)%", re.IGNORECASE)
CORE_REC_RE = re.compile(r"(?:Core\s*Rec(?:all)?|CoreRec(?:all)?|Core\s*Recall)\s*:\s*([0-9.]+)%", re.IGNORECASE)

TEST_LINE_RE = re.compile(
    r"Final Test Loss\s*:\s*([0-9.]+)\s*\|\s*(?:Final Test )?Acc:\s*([0-9.]+)%"
)


def parse_log_file(log_path):
    """Parse epoch history and optional final test metrics from a training log file."""
    data = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_core_fnr": [],
        "val_core_rec": [],
        "lr": [],
        "best_epochs": [],
        "test_loss": None,
        "test_acc": None,
        "test_bal_acc": None,
        "test_f1": None,
        "test_core_fnr": None,
        "test_core_rec": None,
    }
    if not log_path or not os.path.exists(log_path):
        return data

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                m = EPOCH_LINE_RE.match(line)
                if m:
                    ep = int(m.group(1))
                    data["epochs"].append(ep)
                    data["train_loss"].append(float(m.group(2)))
                    data["train_acc"].append(float(m.group(3)))
                    data["val_loss"].append(float(m.group(4)))
                    data["val_acc"].append(float(m.group(5)))
                    data["lr"].append(float(m.group(6)))
                    if m.group(7) or "(*)" in line:
                        data["best_epochs"].append(ep)
                    m_fnr = CORE_FNR_RE.search(line)
                    if m_fnr:
                        data["val_core_fnr"].append(float(m_fnr.group(1)))
                    m_rec = CORE_REC_RE.search(line)
                    if m_rec:
                        data["val_core_rec"].append(float(m_rec.group(1)))
                    continue
                if "Final Test Loss" in line:
                    tm = TEST_LINE_RE.search(line)
                    if tm:
                        data["test_loss"] = float(tm.group(1))
                        data["test_acc"] = float(tm.group(2))
                        m_bal = re.search(r"BalAcc:\s*([0-9.]+)%", line)
                        if m_bal:
                            data["test_bal_acc"] = float(m_bal.group(1))
                        m_f1 = re.search(r"F1:\s*([0-9.]+)", line)
                        if m_f1:
                            data["test_f1"] = float(m_f1.group(1))
                        m_t_fnr = CORE_FNR_RE.search(line)
                        if m_t_fnr:
                            data["test_core_fnr"] = float(m_t_fnr.group(1))
                        m_t_rec = CORE_REC_RE.search(line)
                        if m_t_rec:
                            data["test_core_rec"] = float(m_t_rec.group(1))
    except Exception:
        pass
    return data


def update_live_plots_for_log(log_path):
    """Update both the per-experiment live training plot (PNG + HTML) and the master Study 2 live dashboard PNG."""
    if not log_path or not os.path.exists(log_path):
        return None

    exp_dir = os.path.dirname(os.path.abspath(log_path))
    exp_tag = os.path.basename(exp_dir)

    gcn_logs = sorted(glob.glob(os.path.join(exp_dir, "training_metrics_gcn*.log")))
    gat_logs = sorted(glob.glob(os.path.join(exp_dir, "training_metrics_gat*.log")))

    gcn_data = parse_log_file(gcn_logs[0]) if gcn_logs else parse_log_file("")
    gat_data = parse_log_file(gat_logs[0]) if gat_logs else parse_log_file("")

    if not gcn_data["epochs"] and not gat_data["epochs"]:
        return None

    # Always write interactive Plotly HTML learning curve (pure stdlib json, no matplotlib dependency required)
    try:
        import json
        traces_html = []
        if gcn_data["epochs"]:
            traces_html.extend([
                {"x": gcn_data["epochs"], "y": gcn_data["train_loss"], "mode": "lines+markers", "name": "GCN Train Loss", "line": {"dash": "dash", "color": "#60a5fa"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gcn_data["epochs"], "y": gcn_data["val_loss"], "mode": "lines+markers", "name": f"GCN Val Loss (min={min(gcn_data['val_loss']):.4f})", "line": {"width": 3, "color": "#2563eb"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gcn_data["epochs"], "y": gcn_data["train_acc"], "mode": "lines+markers", "name": "GCN Train Acc (%)", "line": {"dash": "dash", "color": "#60a5fa"}, "xaxis": "x2", "yaxis": "y2"},
                {"x": gcn_data["epochs"], "y": gcn_data["val_acc"], "mode": "lines+markers", "name": f"GCN Val Acc (best={max(gcn_data['val_acc']):.2f}%)", "line": {"width": 3, "color": "#2563eb"}, "xaxis": "x2", "yaxis": "y2"},
            ])
        if gat_data["epochs"]:
            traces_html.extend([
                {"x": gat_data["epochs"], "y": gat_data["train_loss"], "mode": "lines+markers", "name": "GAT Train Loss", "line": {"dash": "dash", "color": "#fb923c"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gat_data["epochs"], "y": gat_data["val_loss"], "mode": "lines+markers", "name": f"GAT Val Loss (min={min(gat_data['val_loss']):.4f})", "line": {"width": 3, "color": "#ea580c"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gat_data["epochs"], "y": gat_data["train_acc"], "mode": "lines+markers", "name": "GAT Train Acc (%)", "line": {"dash": "dash", "color": "#fb923c"}, "xaxis": "x2", "yaxis": "y2"},
                {"x": gat_data["epochs"], "y": gat_data["val_acc"], "mode": "lines+markers", "name": f"GAT Val Acc (best={max(gat_data['val_acc']):.2f}%)", "line": {"width": 3, "color": "#ea580c"}, "xaxis": "x2", "yaxis": "y2"},
            ])
        layout_html = {
            "title": f"Interactive GNN Learning Curves — Loss & Accuracy ({exp_tag})",
            "template": "plotly_white",
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12, "color": "#111827"},
            "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
            "xaxis1": {"title": "Epoch"},
            "yaxis1": {"title": "Binary Cross-Entropy Loss"},
            "xaxis2": {"title": "Epoch"},
            "yaxis2": {"title": "Node Classification Accuracy [%]"},
        }
        out_exp_html = os.path.join(exp_dir, f"live_plots_GNN_{exp_tag}.html")
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Live GNN Learning Curves ({exp_tag})</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{ margin: 0; background: #ffffff; color: #111827; font-family: system-ui, sans-serif; }}
    #plot {{ width: 100vw; height: 96vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const traces = {json.dumps(traces_html)};
    const layout = {json.dumps(layout_html)};
    Plotly.newPlot('plot', traces, layout, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""
        with open(out_exp_html, "w", encoding="utf-8") as hf:
            hf.write(html_doc)
    except Exception:
        pass

    if plt is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)

    if gcn_data["epochs"]:
        ep = gcn_data["epochs"]
        best_val_acc = max(gcn_data["val_acc"])
        min_val_loss = min(gcn_data["val_loss"])
        best_loss_idx = gcn_data["val_loss"].index(min_val_loss)
        best_loss_epoch = ep[best_loss_idx]

        loss_lbl = f"GCN Val Loss (min={min_val_loss:.4f} @ Ep {best_loss_epoch}"
        if gcn_data["val_core_fnr"]:
            loss_lbl += f", min CoreFNR={min(gcn_data['val_core_fnr']):.2f}%"
        loss_lbl += ")"

        ax1.plot(ep, gcn_data["train_loss"], "--", color="#60a5fa", alpha=0.85, lw=1.6, label="GCN Train Loss")
        ax1.plot(ep, gcn_data["val_loss"], "-o", color="#1d4ed8", ms=3.5, lw=2.2, label=loss_lbl)
        ax1.axvline(best_loss_epoch, color="#1d4ed8", linestyle=":", alpha=0.6, lw=1.5)
        ax1.plot(best_loss_epoch, min_val_loss, "*", color="#fbbf24", markeredgecolor="#1d4ed8", ms=12, zorder=5, label=f"GCN Best Checkpoint (Ep {best_loss_epoch})")

        ax2.plot(ep, gcn_data["train_acc"], "--", color="#60a5fa", alpha=0.85, lw=1.6, label="GCN Train Acc")
        lbl_acc = f"GCN Val Acc (best={best_val_acc:.2f}%"
        if gcn_data["test_acc"] is not None:
            lbl_acc += f", test={gcn_data['test_acc']:.2f}%"
        if gcn_data["val_core_rec"]:
            lbl_acc += f", max CoreRec={max(gcn_data['val_core_rec']):.2f}%"
        lbl_acc += ")"
        ax2.plot(ep, gcn_data["val_acc"], "-o", color="#1d4ed8", ms=3.5, lw=2.2, label=lbl_acc)

    if gat_data["epochs"]:
        ep = gat_data["epochs"]
        best_val_acc = max(gat_data["val_acc"])
        min_val_loss = min(gat_data["val_loss"])
        best_loss_idx = gat_data["val_loss"].index(min_val_loss)
        best_loss_epoch = ep[best_loss_idx]

        loss_lbl = f"GAT Val Loss (min={min_val_loss:.4f} @ Ep {best_loss_epoch}"
        if gat_data["val_core_fnr"]:
            loss_lbl += f", min CoreFNR={min(gat_data['val_core_fnr']):.2f}%"
        loss_lbl += ")"

        ax1.plot(ep, gat_data["train_loss"], "--", color="#fb923c", alpha=0.85, lw=1.6, label="GAT Train Loss")
        ax1.plot(ep, gat_data["val_loss"], "-s", color="#ea580c", ms=3.5, lw=2.2, label=loss_lbl)
        ax1.axvline(best_loss_epoch, color="#ea580c", linestyle=":", alpha=0.6, lw=1.5)
        ax1.plot(best_loss_epoch, min_val_loss, "*", color="#fbbf24", markeredgecolor="#ea580c", ms=12, zorder=5, label=f"GAT Best Checkpoint (Ep {best_loss_epoch})")

        ax2.plot(ep, gat_data["train_acc"], "--", color="#fb923c", alpha=0.85, lw=1.6, label="GAT Train Acc")
        lbl_acc = f"GAT Val Acc (best={best_val_acc:.2f}%"
        if gat_data["test_acc"] is not None:
            lbl_acc += f", test={gat_data['test_acc']:.2f}%"
        if gat_data["val_core_rec"]:
            lbl_acc += f", max CoreRec={max(gat_data['val_core_rec']):.2f}%"
        lbl_acc += ")"
        ax2.plot(ep, gat_data["val_acc"], "-s", color="#ea580c", ms=3.5, lw=2.2, label=lbl_acc)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.set_title(f"Live GNN Loss Evolution ({exp_tag})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Node Classification Accuracy [%]")
    ax2.set_title(f"Live GNN Accuracy Evolution ({exp_tag})")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out_exp_png = os.path.join(exp_dir, f"live_plots_GNN_{exp_tag}.png")
    plt.savefig(out_exp_png)
    plt.close(fig)

    # Automatically refresh the RMT candle singular value spectrum if best weights exist in exp_dir
    try:
        try:
            from scripts.generate_study2_figures import plot_spectral_analysis
        except ImportError:
            from generate_study2_figures import plot_spectral_analysis
        gcn_w = sorted(glob.glob(os.path.join(exp_dir, "best_gcn*.pth")))
        gat_w = sorted(glob.glob(os.path.join(exp_dir, "best_gat*.pth")))
        if gcn_w or gat_w:
            in_ch = 4 if "4obs_" in exp_tag else 5
            plot_spectral_analysis(
                gcn_w[0] if gcn_w else "",
                gat_w[0] if gat_w else "",
                exp_dir,
                exp_tag,
                in_channels=in_ch,
            )
    except Exception:
        pass

    update_master_study2_live_plot()
    return out_exp_png


def update_master_study2_live_plot():
    """Aggregate all active/completed Study 2 training logs into a single master live figure."""
    if plt is None:
        return None
    study2_dir = os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2")
    os.makedirs(study2_dir, exist_ok=True)
    all_logs = sorted(glob.glob(os.path.join(study2_dir, "**", "training_metrics_*.log"), recursive=True))
    if not all_logs:
        return None

    parsed_runs = []
    for lg in all_logs:
        if "/legacy/" in lg.replace("\\", "/"):
            continue
        d = parse_log_file(lg)
        if d["epochs"]:
            exp_name = os.path.basename(os.path.dirname(lg))
            model_type = "GAT" if "_gat" in os.path.basename(lg) else "GCN"
            parsed_runs.append((f"{exp_name} ({model_type})", d))

    if not parsed_runs:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    cmap = plt.get_cmap("tab10")

    for idx, (label, d) in enumerate(parsed_runs):
        color = cmap(idx % 10)
        ep = d["epochs"]
        best_val = max(d["val_acc"])
        min_loss = min(d["val_loss"])
        best_idx = d["val_loss"].index(min_loss)
        best_ep = ep[best_idx]
        test_str = f" | Test={d['test_acc']:.2f}%" if d["test_acc"] is not None else ""
        core_str = f" | CoreFNR={min(d['val_core_fnr']):.2f}%" if d["val_core_fnr"] else ""
        ax1.plot(ep, d["train_loss"], "--", color=color, alpha=0.45, lw=1.3)
        ax1.plot(ep, d["val_loss"], "-o", color=color, ms=3, lw=2.0, label=f"{label} [Ep {ep[-1]}]{core_str}")
        ax1.plot(best_ep, min_loss, "*", color=color, ms=9, zorder=4)
        ax2.plot(ep, d["train_acc"], "--", color=color, alpha=0.45, lw=1.3)
        ax2.plot(
            ep,
            d["val_acc"],
            "-o",
            color=color,
            ms=3,
            lw=2.0,
            label=f"{label}: Val={best_val:.2f}%{test_str}",
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation & Train BCE Loss (solid=Val, dashed=Train)")
    ax1.set_title("Study 2 Live Training & Validation Loss Across Runs")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Node Classification Accuracy [%]")
    ax2.set_title("Study 2 Live Node Accuracy Evolution Across Runs")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    master_png = os.path.join(study2_dir, "latest_live_training.png")
    plt.savefig(master_png)
    plt.close(fig)
    return master_png
