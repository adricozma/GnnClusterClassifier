"""Real-time training curve plotter for Study 2 V1 and V2 GNN experiments.

Generates publication-grade live training monitoring figures (`live_plots_GNN_*.png`
and `GNN-jed/trained_models/study2/latest_live_training.png`) at the end of every training
epoch so learning dynamics (BCE / weighted Cross-Entropy loss, node classification accuracy,
best validation checkpoints, and learning rate schedules) can be inspected live during SLURM execution.
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

V2_EPOCH_LINE_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*"
    r"([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.eE+-]+)"
    r"(?:\s*\([0-9.]+s\))?\s*(\*|\(\*\))?"
)
EPOCH_LINE_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.eE+-]+)(?:.*?(\(\*\)))?"
)
CORE_FNR_RE = re.compile(r"(?:Core\s*FNR|CoreFNR|FNR)\s*:\s*([0-9.]+)%", re.IGNORECASE)
CORE_REC_RE = re.compile(r"(?:Core\s*Rec(?:all)?|CoreRec(?:all)?|Core\s*Recall)\s*:\s*([0-9.]+)%", re.IGNORECASE)

TEST_LINE_RE = re.compile(
    r"(?:Final\s+)?Test Loss\s*:\s*([0-9.]+)\s*\|\s*(?:Final Test )?Acc:\s*([0-9.]+)%"
)


def parse_log_file(log_path):
    """Parse epoch history and optional final test metrics from a V1 or V2 training log file."""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_core_fnr = []
    val_core_rec = []
    va_bound_f1 = []
    va_pe_rec = []
    va_field_fpr = []
    lr = []
    best_epochs = []

    data = {
        "epochs": epochs,
        "train_loss": train_loss,
        "tr_loss": train_loss,
        "train_acc": train_acc,
        "tr_acc": train_acc,
        "val_loss": val_loss,
        "va_loss": val_loss,
        "val_acc": val_acc,
        "va_acc": val_acc,
        "val_core_fnr": val_core_fnr,
        "va_core_fnr": val_core_fnr,
        "val_core_rec": val_core_rec,
        "va_bound_f1": va_bound_f1,
        "va_pe_rec": va_pe_rec,
        "va_field_fpr": va_field_fpr,
        "lr": lr,
        "best_epochs": best_epochs,
        "best_epoch": None,
        "test_loss": None,
        "test_acc": None,
        "test_bal_acc": None,
        "test_f1": None,
        "test_macro_f1": None,
        "test_bound_f1": None,
        "test_pe_rec": None,
        "test_core_fnr": None,
        "test_core_rec": None,
        "test_field_fpr": None,
        "is_v2": False,
    }
    if not log_path or not os.path.exists(log_path):
        return data

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Study 2 V2" in line or "BoundF1" in line or "PE_Rec" in line or "FieldFPR" in line:
                    data["is_v2"] = True

                m2 = V2_EPOCH_LINE_RE.match(line)
                if m2:
                    data["is_v2"] = True
                    ep = int(m2.group(1))
                    epochs.append(ep)
                    train_loss.append(float(m2.group(2)))
                    train_acc.append(float(m2.group(3)))
                    val_loss.append(float(m2.group(4)))
                    val_acc.append(float(m2.group(5)))
                    va_bound_f1.append(float(m2.group(6)))
                    va_pe_rec.append(float(m2.group(7)))
                    core_fnr = float(m2.group(8))
                    val_core_fnr.append(core_fnr)
                    val_core_rec.append(max(0.0, 100.0 - core_fnr))
                    va_field_fpr.append(float(m2.group(9)))
                    lr.append(float(m2.group(10)))
                    if m2.group(11) or line.rstrip().endswith("*"):
                        best_epochs.append(ep)
                    continue

                m = EPOCH_LINE_RE.match(line)
                if m:
                    ep = int(m.group(1))
                    epochs.append(ep)
                    train_loss.append(float(m.group(2)))
                    train_acc.append(float(m.group(3)))
                    val_loss.append(float(m.group(4)))
                    val_acc.append(float(m.group(5)))
                    lr.append(float(m.group(6)))
                    if m.group(7) or "(*)" in line:
                        best_epochs.append(ep)
                    m_fnr = CORE_FNR_RE.search(line)
                    if m_fnr:
                        val_core_fnr.append(float(m_fnr.group(1)))
                    m_rec = CORE_REC_RE.search(line)
                    if m_rec:
                        val_core_rec.append(float(m_rec.group(1)))
                    continue

                if "Test Loss" in line:
                    tm = TEST_LINE_RE.search(line)
                    if tm:
                        data["test_loss"] = float(tm.group(1))
                        data["test_acc"] = float(tm.group(2))
                        m_bal = re.search(r"BalAcc:\s*([0-9.]+)%", line)
                        if m_bal:
                            data["test_bal_acc"] = float(m_bal.group(1))
                        m_macro_f1 = re.search(r"MacroF1:\s*([0-9.]+)", line)
                        if m_macro_f1:
                            data["test_macro_f1"] = float(m_macro_f1.group(1))
                            data["is_v2"] = True
                        m_bound_f1 = re.search(r"BoundF1:\s*([0-9.]+)", line)
                        if m_bound_f1:
                            data["test_bound_f1"] = float(m_bound_f1.group(1))
                            data["is_v2"] = True
                        m_f1 = re.search(r"(?:^|\|\s*)F1:\s*([0-9.]+)", line)
                        if m_f1:
                            f1_val = float(m_f1.group(1))
                            data["test_f1"] = f1_val
                            if data["is_v2"] and data["test_bound_f1"] is None:
                                data["test_bound_f1"] = f1_val
                        if data["test_bound_f1"] is not None:
                            data["test_f1"] = data["test_bound_f1"]
                        elif data["test_f1"] is None and data["test_macro_f1"] is not None:
                            data["test_f1"] = data["test_macro_f1"]

                        m_pe_rec = re.search(r"PE_Rec:\s*([0-9.]+)%", line)
                        if m_pe_rec:
                            data["test_pe_rec"] = float(m_pe_rec.group(1))
                            data["is_v2"] = True
                        m_field_fpr = re.search(r"FieldFPR:\s*([0-9.]+)%", line)
                        if m_field_fpr:
                            data["test_field_fpr"] = float(m_field_fpr.group(1))
                            data["is_v2"] = True

                        m_t_fnr = CORE_FNR_RE.search(line)
                        if m_t_fnr:
                            data["test_core_fnr"] = float(m_t_fnr.group(1))
                        m_t_rec = CORE_REC_RE.search(line)
                        if m_t_rec:
                            data["test_core_rec"] = float(m_t_rec.group(1))
                        elif data["test_core_fnr"] is not None:
                            data["test_core_rec"] = max(0.0, 100.0 - data["test_core_fnr"])
    except Exception:
        pass

    if best_epochs:
        data["best_epoch"] = best_epochs[-1]
    elif val_loss and epochs:
        data["best_epoch"] = epochs[val_loss.index(min(val_loss))]
    return data


def _infer_log_arch(log_path, is_v2=False):
    """Infer architecture label (GAT/GCN or GAT_V2/GCN_V2) from log filename, directory, or header."""
    fname = os.path.basename(log_path).lower()
    parent = os.path.basename(os.path.dirname(log_path)).lower()
    is_gcn = "_gcn" in fname or fname.startswith("gcn") or parent.endswith("_gcn") or "_gcn_" in parent
    if not is_gcn and os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(512)
            if "GCN_V2" in head or "GCN" in head and "GAT" not in head:
                is_gcn = True
        except Exception:
            pass
    base_arch = "GCN" if is_gcn else "GAT"
    return (base_arch, f"{base_arch}_V2" if is_v2 else base_arch)


def discover_study2_logs():
    """Discover all Study 2 V1 and V2 training logs across study2/ and study2_v2/ without key collisions."""
    study2_dir = os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2")
    study2_v2_dir = os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2_v2")
    entries = []
    seen_paths = set()

    for study_tag, base_dir in (("v1", study2_dir), ("v2", study2_v2_dir)):
        if not os.path.isdir(base_dir):
            continue
        all_logs = sorted(glob.glob(os.path.join(base_dir, "**", "training_metrics_*.log"), recursive=True))
        for lg in all_logs:
            norm_lg = os.path.abspath(lg)
            if norm_lg in seen_paths or "/legacy/" in norm_lg.replace("\\", "/"):
                continue
            seen_paths.add(norm_lg)

            rel_dir = os.path.relpath(os.path.dirname(norm_lg), os.path.abspath(base_dir)).replace("\\", "/")
            fname = os.path.basename(norm_lg)
            if rel_dir == ".":
                exp_name = os.path.splitext(fname)[0].replace("training_metrics_", "")
            else:
                exp_name = rel_dir

            is_v2 = (
                study_tag == "v2"
                or exp_name.startswith("v2_")
                or exp_name.startswith("v2/")
                or "_v2_" in fname.lower()
            )
            if study_tag == "v2":
                exp_key = exp_name if exp_name.startswith("v2/") else f"v2/{exp_name}"
            else:
                exp_key = exp_name

            base_arch, arch_label = _infer_log_arch(norm_lg, is_v2=is_v2)
            entries.append({
                "log_path": norm_lg,
                "study": "v2" if is_v2 else "v1",
                "exp_key": exp_key,
                "exp_name": exp_name,
                "arch": base_arch,
                "arch_label": arch_label,
                "label": f"{exp_key} ({arch_label})",
            })
    return entries


def update_live_plots_for_log(log_path):
    """Update both the per-experiment live training plot (PNG + HTML) and the master Study 2 live dashboard PNG."""
    if not log_path or not os.path.exists(log_path):
        return None

    exp_dir = os.path.dirname(os.path.abspath(log_path))
    exp_tag = os.path.basename(exp_dir)

    all_exp_logs = sorted(glob.glob(os.path.join(exp_dir, "training_metrics_*.log")))
    gcn_logs = []
    gat_logs = []
    for lg in all_exp_logs:
        base_arch, _ = _infer_log_arch(lg)
        if base_arch == "GCN":
            gcn_logs.append(lg)
        else:
            gat_logs.append(lg)

    gcn_data = parse_log_file(gcn_logs[0]) if gcn_logs else parse_log_file("")
    gat_data = parse_log_file(gat_logs[0]) if gat_logs else parse_log_file("")

    if not gcn_data["epochs"] and not gat_data["epochs"]:
        return None

    is_v2_exp = bool(gcn_data.get("is_v2") or gat_data.get("is_v2") or "study2_v2" in exp_dir.replace("\\", "/"))
    loss_ylabel = "Cross-Entropy Loss (3-Class)" if is_v2_exp else "Binary Cross-Entropy Loss"
    gcn_prefix = "GCN_V2" if gcn_data.get("is_v2") else "GCN"
    gat_prefix = "GAT_V2" if gat_data.get("is_v2") else "GAT"

    # Always write interactive Plotly HTML learning curve (pure stdlib json, no matplotlib dependency required)
    try:
        import json
        traces_html = []
        if gcn_data["epochs"]:
            traces_html.extend([
                {"x": gcn_data["epochs"], "y": gcn_data["train_loss"], "mode": "lines+markers", "name": f"{gcn_prefix} Train Loss", "line": {"dash": "dash", "color": "#60a5fa"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gcn_data["epochs"], "y": gcn_data["val_loss"], "mode": "lines+markers", "name": f"{gcn_prefix} Val Loss (min={min(gcn_data['val_loss']):.4f})", "line": {"width": 3, "color": "#2563eb"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gcn_data["epochs"], "y": gcn_data["train_acc"], "mode": "lines+markers", "name": f"{gcn_prefix} Train Acc (%)", "line": {"dash": "dash", "color": "#60a5fa"}, "xaxis": "x2", "yaxis": "y2"},
                {"x": gcn_data["epochs"], "y": gcn_data["val_acc"], "mode": "lines+markers", "name": f"{gcn_prefix} Val Acc (best={max(gcn_data['val_acc']):.2f}%)", "line": {"width": 3, "color": "#2563eb"}, "xaxis": "x2", "yaxis": "y2"},
            ])
        if gat_data["epochs"]:
            traces_html.extend([
                {"x": gat_data["epochs"], "y": gat_data["train_loss"], "mode": "lines+markers", "name": f"{gat_prefix} Train Loss", "line": {"dash": "dash", "color": "#fb923c"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gat_data["epochs"], "y": gat_data["val_loss"], "mode": "lines+markers", "name": f"{gat_prefix} Val Loss (min={min(gat_data['val_loss']):.4f})", "line": {"width": 3, "color": "#ea580c"}, "xaxis": "x1", "yaxis": "y1"},
                {"x": gat_data["epochs"], "y": gat_data["train_acc"], "mode": "lines+markers", "name": f"{gat_prefix} Train Acc (%)", "line": {"dash": "dash", "color": "#fb923c"}, "xaxis": "x2", "yaxis": "y2"},
                {"x": gat_data["epochs"], "y": gat_data["val_acc"], "mode": "lines+markers", "name": f"{gat_prefix} Val Acc (best={max(gat_data['val_acc']):.2f}%)", "line": {"width": 3, "color": "#ea580c"}, "xaxis": "x2", "yaxis": "y2"},
            ])
        layout_html = {
            "title": f"Interactive GNN Learning Curves — Loss & Accuracy ({exp_tag})",
            "template": "plotly_white",
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12, "color": "#111827"},
            "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
            "xaxis1": {"title": "Epoch"},
            "yaxis1": {"title": loss_ylabel},
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

    def _plot_series(ax_l, ax_a, mdata, mprefix, tr_color, va_color, marker):
        if not mdata["epochs"]:
            return
        ep = mdata["epochs"]
        best_val_acc = max(mdata["val_acc"])
        min_val_loss = min(mdata["val_loss"])
        best_loss_idx = mdata["val_loss"].index(min_val_loss)
        best_loss_epoch = ep[best_loss_idx]

        loss_lbl = f"{mprefix} Val Loss (min={min_val_loss:.4f} @ Ep {best_loss_epoch}"
        if mdata["val_core_fnr"]:
            loss_lbl += f", min CoreFNR={min(mdata['val_core_fnr']):.2f}%"
        if mdata["va_bound_f1"]:
            loss_lbl += f", max BoundF1={max(mdata['va_bound_f1']):.3f}"
        loss_lbl += ")"

        ax_l.plot(ep, mdata["train_loss"], "--", color=tr_color, alpha=0.85, lw=1.6, label=f"{mprefix} Train Loss")
        ax_l.plot(ep, mdata["val_loss"], f"-{marker}", color=va_color, ms=3.5, lw=2.2, label=loss_lbl)
        ax_l.axvline(best_loss_epoch, color=va_color, linestyle=":", alpha=0.6, lw=1.5)
        ax_l.plot(best_loss_epoch, min_val_loss, "*", color="#fbbf24", markeredgecolor=va_color, ms=12, zorder=5, label=f"{mprefix} Best Checkpoint (Ep {best_loss_epoch})")

        ax_a.plot(ep, mdata["train_acc"], "--", color=tr_color, alpha=0.85, lw=1.6, label=f"{mprefix} Train Acc")
        lbl_acc = f"{mprefix} Val Acc (best={best_val_acc:.2f}%"
        if mdata["test_acc"] is not None:
            lbl_acc += f", test={mdata['test_acc']:.2f}%"
        if mdata["val_core_rec"]:
            lbl_acc += f", max CoreRec={max(mdata['val_core_rec']):.2f}%"
        lbl_acc += ")"
        ax_a.plot(ep, mdata["val_acc"], f"-{marker}", color=va_color, ms=3.5, lw=2.2, label=lbl_acc)

    _plot_series(ax1, ax2, gcn_data, gcn_prefix, "#60a5fa", "#1d4ed8", "o")
    _plot_series(ax1, ax2, gat_data, gat_prefix, "#fb923c", "#ea580c", "s")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(loss_ylabel)
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

    # Automatically refresh the RMT candle singular value spectrum if best V1 weights exist in exp_dir
    if not is_v2_exp:
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
    """Aggregate all active/completed Study 2 V1 and V2 training logs into a single master live figure."""
    if plt is None:
        return None
    study2_dir = os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2")
    os.makedirs(study2_dir, exist_ok=True)
    discovered = discover_study2_logs()
    if not discovered:
        return None

    parsed_runs = []
    for entry in discovered:
        d = parse_log_file(entry["log_path"])
        if d["epochs"]:
            arch_lbl = f"{entry['arch']}_V2" if (d.get("is_v2") and "_V2" not in entry["arch_label"]) else entry["arch_label"]
            label = f"{entry['exp_key']} ({arch_lbl})"
            parsed_runs.append((label, d))

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
        f1_str = f" | BoundF1={max(d['va_bound_f1']):.3f}" if d.get("va_bound_f1") else ""
        ax1.plot(ep, d["train_loss"], "--", color=color, alpha=0.45, lw=1.3)
        ax1.plot(ep, d["val_loss"], "-o", color=color, ms=3, lw=2.0, label=f"{label} [Ep {ep[-1]}]{core_str}{f1_str}")
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
    ax1.set_ylabel("Validation & Train Loss (solid=Val, dashed=Train)")
    ax1.set_title("Study 2 (V1 & V2) Live Training & Validation Loss")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8.0)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Node Classification Accuracy [%]")
    ax2.set_title("Study 2 (V1 & V2) Live Node Accuracy Evolution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8.0, loc="lower right")

    plt.tight_layout()
    master_png = os.path.join(study2_dir, "latest_live_training.png")
    plt.savefig(master_png)
    plt.close(fig)
    return master_png
