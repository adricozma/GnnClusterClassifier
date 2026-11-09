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
    r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.eE+-]+)(\s*\(\*\))?"
)
TEST_LINE_RE = re.compile(
    r"Final Test Loss\s*:\s*([0-9.]+)\s*\|\s*Acc:\s*([0-9.]+)%(?:\s*\|\s*BalAcc:\s*([0-9.]+)%\s*\|\s*F1:\s*([0-9.]+))?"
)


def parse_log_file(log_path):
    """Parse epoch history and optional final test metrics from a training log file."""
    data = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "best_epochs": [],
        "test_loss": None,
        "test_acc": None,
        "test_bal_acc": None,
        "test_f1": None,
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
                    if m.group(7):
                        data["best_epochs"].append(ep)
                    continue
                tm = TEST_LINE_RE.search(line)
                if tm:
                    data["test_loss"] = float(tm.group(1))
                    data["test_acc"] = float(tm.group(2))
                    if tm.group(3):
                        data["test_bal_acc"] = float(tm.group(3))
                    if tm.group(4):
                        data["test_f1"] = float(tm.group(4))
    except Exception:
        pass
    return data


def update_live_plots_for_log(log_path):
    """Update both the per-experiment live training plot and the master Study 2 live dashboard PNG."""
    if plt is None or not log_path or not os.path.exists(log_path):
        return None

    exp_dir = os.path.dirname(os.path.abspath(log_path))
    exp_tag = os.path.basename(exp_dir)

    gcn_logs = sorted(glob.glob(os.path.join(exp_dir, "training_metrics_gcn*.log")))
    gat_logs = sorted(glob.glob(os.path.join(exp_dir, "training_metrics_gat*.log")))

    gcn_data = parse_log_file(gcn_logs[0]) if gcn_logs else parse_log_file("")
    gat_data = parse_log_file(gat_logs[0]) if gat_logs else parse_log_file("")

    if not gcn_data["epochs"] and not gat_data["epochs"]:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)

    if gcn_data["epochs"]:
        ep = gcn_data["epochs"]
        best_val_acc = max(gcn_data["val_acc"])
        ax1.plot(ep, gcn_data["train_loss"], "--", color="#60a5fa", alpha=0.85, lw=1.6, label="GCN Train Loss")
        ax1.plot(ep, gcn_data["val_loss"], "-o", color="#1d4ed8", ms=3.5, lw=2.2, label=f"GCN Val Loss (min={min(gcn_data['val_loss']):.4f})")
        ax2.plot(ep, gcn_data["train_acc"], "--", color="#60a5fa", alpha=0.85, lw=1.6, label="GCN Train Acc")
        lbl_acc = f"GCN Val Acc (best={best_val_acc:.2f}%"
        if gcn_data["test_acc"] is not None:
            lbl_acc += f", test={gcn_data['test_acc']:.2f}%"
        lbl_acc += ")"
        ax2.plot(ep, gcn_data["val_acc"], "-o", color="#1d4ed8", ms=3.5, lw=2.2, label=lbl_acc)

    if gat_data["epochs"]:
        ep = gat_data["epochs"]
        best_val_acc = max(gat_data["val_acc"])
        ax1.plot(ep, gat_data["train_loss"], "--", color="#fb923c", alpha=0.85, lw=1.6, label="GAT Train Loss")
        ax1.plot(ep, gat_data["val_loss"], "-s", color="#ea580c", ms=3.5, lw=2.2, label=f"GAT Val Loss (min={min(gat_data['val_loss']):.4f})")
        ax2.plot(ep, gat_data["train_acc"], "--", color="#fb923c", alpha=0.85, lw=1.6, label="GAT Train Acc")
        lbl_acc = f"GAT Val Acc (best={best_val_acc:.2f}%"
        if gat_data["test_acc"] is not None:
            lbl_acc += f", test={gat_data['test_acc']:.2f}%"
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
        test_str = f" | Test={d['test_acc']:.2f}%" if d["test_acc"] is not None else ""
        ax1.plot(ep, d["train_loss"], "--", color=color, alpha=0.55, lw=1.3)
        ax1.plot(ep, d["val_loss"], "-o", color=color, ms=3, lw=2.0, label=f"{label} [Ep {ep[-1]}]")
        ax2.plot(ep, d["train_acc"], "--", color=color, alpha=0.55, lw=1.3)
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
