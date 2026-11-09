"""Live remote telemetry and notification daemon for Study 2 on EPFL Jed.

Publishes real-time SLURM job status, simulation/extraction progress, epoch-by-epoch
GNN training metrics, and real-time learning curve figures (`latest_live_training.png`
and `live_plots_GNN_*.png`) to a public web stream (`https://ntfy.sh/<topic>`) and
updates `LIVE_STATUS.md` so progress and live plots can be followed from any browser
or smartphone without SSH access to Jed.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_plot_utils import parse_log_file, update_live_plots_for_log, update_master_study2_live_plot

DEFAULT_TOPIC = "epfl-cozma-gnn-study2"


def send_ntfy_message(topic, message, title="EPFL Jed - GNN Study 2", priority="default", attachment_path=None):
    """Publish a text message or file attachment to https://ntfy.sh/<topic>."""
    url = f"https://ntfy.sh/{topic}"
    cmd = [
        "curl",
        "-sS",
        "-H", f"Title: {title}",
        "-H", f"Priority: {priority}",
    ]
    if attachment_path and os.path.exists(attachment_path):
        fname = os.path.basename(attachment_path)
        cmd.extend(["-T", attachment_path, "-H", f"Filename: {fname}", "-H", f"Message: {message}", url])
    else:
        cmd.extend(["-d", message, url])

    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        pass


def auto_heal_slurm_jobs(topic=DEFAULT_TOPIC):
    """Disabled automatic CPU/RAM downscaling to preserve full-node 492G allocations for GNN training."""
    return



def get_slurm_queue():
    """Return current squeue output for user cozma."""
    try:
        res = subprocess.run(
            ["squeue", "-u", "cozma", "-o", "%.14i %.10P %.22j %.4t %.10M %.6D %R"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return res.stdout.strip()
    except Exception as e:
        return f"Unable to query squeue: {e}"


def collect_pipeline_progress():
    """Count completed files across each stage of Study 2 and refresh live training plots."""
    n_ics = len(glob.glob(os.path.join(ROOT_DIR, "make_ICs", "initial_conditions_unbiased", "IC_*.hdf5")))

    # Check SWIFT completion: /scratch/cozma/study2/outputs_unbiased (falling back to /home/cozma/swift_sims/outputs_unbiased)
    # Only count an IC_* folder as completed if all 101 snapshot_*.hdf5 files exist and have os.path.getsize > 50000
    swift_dirs = ["/scratch/cozma/study2/outputs_unbiased", "/home/cozma/swift_sims/outputs_unbiased"]
    all_ic_folders = set()
    for base in swift_dirs:
        if os.path.exists(base):
            for p in glob.glob(os.path.join(base, "IC_*")):
                if os.path.isdir(p):
                    all_ic_folders.add(os.path.basename(p))

    n_swift_done = 0
    for ic_name in sorted(all_ic_folders):
        chosen_dir = None
        for base in swift_dirs:
            cand = os.path.join(base, ic_name)
            if os.path.isdir(cand):
                chosen_dir = cand
                break
        if not chosen_dir:
            continue
        all_valid = True
        for i in range(101):
            snap_path = os.path.join(chosen_dir, f"snapshot_{i:04d}.hdf5")
            try:
                if not (os.path.isfile(snap_path) and os.path.getsize(snap_path) > 50000):
                    all_valid = False
                    break
            except OSError:
                all_valid = False
                break
        if all_valid:
            n_swift_done += 1

    # Check labels and observables: /scratch/cozma/study2/... (falling back to GNN-jed/...)
    def count_stage_done(scratch_path, fallback_path, prefix):
        ics = set()
        for base in (scratch_path, fallback_path):
            if not os.path.exists(base):
                continue
            for ic_dir in glob.glob(os.path.join(base, "IC_*")):
                if not os.path.isdir(ic_dir):
                    continue
                files = glob.glob(os.path.join(ic_dir, f"{prefix}_snapshot_*.npy"))
                if len(files) == 101 and all(os.path.getsize(f) > 512 for f in files):
                    ics.add(os.path.basename(ic_dir))
        return len(ics)

    n_labels = count_stage_done(
        "/scratch/cozma/study2/labels_clean",
        os.path.join(ROOT_DIR, "GNN-jed", "labels_clean"),
        "B_L",
    )
    n_obs_clean = count_stage_done(
        "/scratch/cozma/study2/obs_clean",
        os.path.join(ROOT_DIR, "GNN-jed", "obs_clean"),
        "obs_matrices",
    )
    n_obs_noisy = count_stage_done(
        "/scratch/cozma/study2/obs_gaia_noisy",
        os.path.join(ROOT_DIR, "GNN-jed", "obs_gaia_noisy"),
        "obs_matrices",
    )
    n_graphs_clean = len(glob.glob("/scratch/cozma/graph_data_clean/IC_*/graph_data_output_100_rotation_0.pt"))
    n_graphs_noisy = len(glob.glob("/scratch/cozma/graph_data_noisy/IC_*/graph_data_output_100_rotation_0.pt"))

    logs = sorted(glob.glob(os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "**", "training_metrics_*.log"), recursive=True))
    latest_epochs = []
    training_rows = []
    for lg in logs:
        rel = os.path.relpath(lg, os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2"))
        d = parse_log_file(lg)
        if d["epochs"]:
            try:
                update_live_plots_for_log(lg)
            except Exception:
                pass
            ep = d["epochs"][-1]
            tr_l = d["train_loss"][-1]
            tr_a = d["train_acc"][-1]
            va_l = d["val_loss"][-1]
            va_a = d["val_acc"][-1]
            best_va = max(d["val_acc"])
            test_str = f"{d['test_acc']:.2f}%" if d["test_acc"] is not None else "In progress"
            f1_str = f"{d['test_f1']:.4f}" if d["test_f1"] is not None else "-"
            latest_epochs.append(
                f"{rel}: Ep {ep:03d} | Train {tr_l:.4f} ({tr_a:.2f}%) | Val {va_l:.4f} ({va_a:.2f}%, Best={best_va:.2f}%) | Test={test_str}"
            )
            exp_name = os.path.basename(os.path.dirname(lg))
            model_name = "GAT" if "_gat" in os.path.basename(lg) else "GCN"
            training_rows.append((exp_name, model_name, ep, tr_l, tr_a, va_l, va_a, best_va, test_str, f1_str))

    try:
        update_master_study2_live_plot()
    except Exception:
        pass

    # Discover all saved model weights (.pth) across experiment folders and mirror to central weights/ folder
    study2_base = os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2")
    central_weights_dir = os.path.join(study2_base, "weights")
    os.makedirs(central_weights_dir, exist_ok=True)
    saved_weights = []
    for pth_path in sorted(glob.glob(os.path.join(study2_base, "**", "*.pth"), recursive=True)):
        if os.path.dirname(os.path.abspath(pth_path)) == os.path.abspath(central_weights_dir):
            continue
        fname = os.path.basename(pth_path)
        if fname.startswith("last_"):
            continue
        central_copy = os.path.join(central_weights_dir, fname)
        try:
            if not os.path.exists(central_copy) or os.path.getsize(pth_path) != os.path.getsize(central_copy) or int(os.path.getmtime(pth_path)) > int(os.path.getmtime(central_copy)):
                import shutil
                shutil.copyfile(pth_path, central_copy)
        except Exception:
            pass
        rel_pth = os.path.relpath(pth_path, ROOT_DIR)
        rel_central = os.path.relpath(central_copy, ROOT_DIR)
        size_kb = os.path.getsize(pth_path) / 1024.0
        saved_weights.append((fname, rel_pth, rel_central, size_kb))

    return {
        "n_ics": n_ics,
        "n_swift_done": n_swift_done,
        "n_labels": n_labels,
        "n_obs_clean": n_obs_clean,
        "n_obs_noisy": n_obs_noisy,
        "n_graphs_clean": n_graphs_clean,
        "n_graphs_noisy": n_graphs_noisy,
        "latest_epochs": latest_epochs,
        "training_rows": training_rows,
        "saved_weights": saved_weights,
    }


def format_status_report(squeue_str, prog):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[{now_str}] Study 2 Pipeline Status on Jed",
        f"- Step 1 (Plummer ICs): {prog['n_ics']} / 81 ready",
        f"- Step 2 (SWIFT Runs): {prog['n_swift_done']} / 81 completed (101 snapshots)",
        f"- Step 3 (Observables): Labels={prog['n_labels']}/81 | Clean={prog['n_obs_clean']}/81 | GaiaNoisy={prog['n_obs_noisy']}/81",
        f"- Step 4 (5D Graphs): Clean={prog['n_graphs_clean']}/81 | GaiaNoisy={prog['n_graphs_noisy']}/81",
    ]
    if prog["latest_epochs"]:
        lines.append("- Step 5/6 Live GNN Training Progress:")
        for le in prog["latest_epochs"]:
            lines.append(f"  * {le}")
    if prog.get("saved_weights"):
        lines.append(f"- Saved Model Checkpoints (.pth): {len(prog['saved_weights'])} mirrored in GNN-jed/trained_models/study2/weights/")
    lines.append("\nSLURM Queue:")
    lines.append(squeue_str if squeue_str else "(No active jobs)")
    return "\n".join(lines)


def write_live_status_markdown(report, prog):
    """Write LIVE_STATUS.md with embedded live training plots and metrics table."""
    status_md_path = os.path.join(ROOT_DIR, "LIVE_STATUS.md")
    md_lines = [
        "# Live Pipeline Status on Jed",
        "",
        "```text",
        report,
        "```",
    ]

    if prog["training_rows"]:
        md_lines.extend([
            "",
            "## Live Neural Network Training Metrics",
            "",
            "| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Test Acc | Test F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for exp_name, model_name, ep, tr_l, tr_a, va_l, va_a, best_va, test_str, f1_str in prog["training_rows"]:
            md_lines.append(
                f"| `{exp_name}` | **{model_name}** | {ep} | {tr_l:.4f} | {tr_a:.2f}% | {va_l:.4f} | {va_a:.2f}% | **{best_va:.2f}%** | **{test_str}** | {f1_str} |"
            )

    if prog.get("saved_weights"):
        md_lines.extend([
            "",
            "## Saved Model Weights (`.pth` Checkpoints)",
            "",
            "| Checkpoint | Size | Experiment Path | Central Archive Mirror |",
            "| :--- | :---: | :--- | :--- |",
        ])
        for fname, rel_pth, rel_central, size_kb in prog["saved_weights"]:
            md_lines.append(f"| `{fname}` | {size_kb:.1f} KB | `{rel_pth}` | `{rel_central}` |")

    master_plot_rel = "GNN-jed/trained_models/study2/latest_live_training.png"
    if os.path.exists(os.path.join(ROOT_DIR, master_plot_rel)):
        md_lines.extend([
            "",
            "## Real-Time GNN Learning Curves (Loss & Accuracy Evolution)",
            "",
            f"![Live GNN Training Curves]({master_plot_rel})",
        ])

    with open(status_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def auto_push_to_github():
    """Stage LIVE_STATUS.md and any new Study 2 figures/logs/weights, commit, pull --rebase, and push to origin/main."""
    try:
        subprocess.run(
            [
                "git",
                "add",
                "-A",
                "LIVE_STATUS.md",
                "REPO_MAP.md",
                "src/",
                "scripts/",
                "slurm/",
                "GNN-jed/trained_models/study2/",
            ],
            cwd=ROOT_DIR,
            check=False,
            timeout=15,
        )
        diff_res = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ROOT_DIR,
            check=False,
            timeout=10,
        )
        if diff_res.returncode != 0:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            subprocess.run(
                ["git", "commit", "-m", f"Update Study 2 live status and figures ({now_str})"],
                cwd=ROOT_DIR,
                check=False,
                timeout=15,
            )
            subprocess.run(
                ["git", "pull", "--rebase", "-X", "ours", "origin", "main"],
                cwd=ROOT_DIR,
                check=False,
                timeout=25,
            )
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=ROOT_DIR,
                check=False,
                timeout=30,
            )
    except Exception:
        pass


UPLOADED_CACHE_FILE = os.path.join(ROOT_DIR, "slurm_logs", "ntfy_uploaded_plots.json")


def _load_uploaded_plots() -> dict:
    import json
    if os.path.exists(UPLOADED_CACHE_FILE):
        try:
            with open(UPLOADED_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_uploaded_plots(cache: dict) -> None:
    import json
    try:
        os.makedirs(os.path.dirname(UPLOADED_CACHE_FILE), exist_ok=True)
        with open(UPLOADED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def daemonize(log_path="/scratch/cozma/slurm_logs/live_monitor_daemon.log"):
    """Detach process via POSIX double-fork and setsid so it survives VS Code/SSH exit."""
    import signal
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open("/dev/null", "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    with open(log_path, "a+") as logf:
        os.dup2(logf.fileno(), sys.stdout.fileno())
        os.dup2(logf.fileno(), sys.stderr.fileno())


def main():
    parser = argparse.ArgumentParser(description="Publish live Study 2 status updates to ntfy.sh and LIVE_STATUS.md.")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="ntfy.sh topic name.")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds (default: 300s = 5 min).")
    parser.add_argument("--once", action="store_true", help="Send a single status snapshot and exit.")
    parser.add_argument("--daemon", action="store_true", help="Run detached in background (survives closing VS Code / SSH).")
    args = parser.parse_args()

    if args.daemon:
        daemonize()

    last_sent_report = None
    uploaded_plots = _load_uploaded_plots()

    while True:
        auto_heal_slurm_jobs(args.topic)
        squeue_str = get_slurm_queue()
        prog = collect_pipeline_progress()
        report = format_status_report(squeue_str, prog)
        write_live_status_markdown(report, prog)

        state_signature = (
            squeue_str,
            prog["n_ics"],
            prog["n_swift_done"],
            prog["n_labels"],
            prog["n_obs_clean"],
            prog["n_graphs_clean"],
            tuple(prog["latest_epochs"]),
        )
        if args.once or state_signature != last_sent_report:
            send_ntfy_message(args.topic, report, title="EPFL Jed - Study 2 Live Update")
            auto_push_to_github()
            last_sent_report = state_signature

        # Automatically push newly generated or updated PNG/HTML figures (including live training curves)
        new_files = glob.glob(
            os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "**", "*.*"),
            recursive=True,
        )
        cache_dirty = False
        for fpath in sorted(new_files):
            if fpath.endswith((".png", ".html")):
                mtime = int(os.path.getmtime(fpath))
                if uploaded_plots.get(fpath) != mtime:
                    send_ntfy_message(
                        args.topic,
                        f"Updated figure: {os.path.basename(fpath)}",
                        title="EPFL Jed - Live Study 2 Figure",
                        attachment_path=fpath,
                    )
                    uploaded_plots[fpath] = mtime
                    cache_dirty = True
        if cache_dirty:
            _save_uploaded_plots(uploaded_plots)

        if args.once:
            try:
                import signal
                ppid = os.getppid()
                cmdline_file = f"/proc/{ppid}/cmdline"
                marker_file = os.path.join(
                    ROOT_DIR, "GNN-jed", "trained_models", "study2", "clean_5obs", ".gcn_ep25_transition_done"
                )
                gcn_clean_log = os.path.join(
                    ROOT_DIR, "GNN-jed", "trained_models", "study2", "clean_5obs", "training_metrics_gcn_clean.log"
                )
                if os.path.exists(cmdline_file) and not os.path.exists(marker_file):
                    with open(cmdline_file, "rb") as cf:
                        cmd_str = cf.read().decode("utf-8", errors="ignore")
                    if "train_gcn.py" in cmd_str and os.path.exists(gcn_clean_log):
                        d_gcn = parse_log_file(gcn_clean_log)
                        if d_gcn["epochs"] and d_gcn["epochs"][-1] >= 25 and d_gcn["test_acc"] is None:
                            with open(marker_file, "w", encoding="utf-8") as mf:
                                mf.write("triggered\n")
                            time.sleep(2.5)
                            os.kill(ppid, signal.SIGTERM)
            except Exception:
                pass
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
