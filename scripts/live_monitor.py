"""Live remote telemetry and notification daemon for Study 2 on EPFL Jed.

Publishes real-time SLURM job status, simulation/extraction progress, epoch-by-epoch
GNN training metrics, and generated PNG/HTML figures to a public web stream
(`https://ntfy.sh/<topic>`) and updates `LIVE_STATUS.md` so progress can be followed
from any browser or smartphone without SSH access to Jed.
"""

import argparse
import glob
import os
import subprocess
import time
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


def get_slurm_queue():
    """Return current squeue output for user cozma."""
    try:
        res = subprocess.run(
            ["squeue", "-u", "cozma", "-o", "%.12i %.10P %.22j %.4t %.10M %.6D %R"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return res.stdout.strip()
    except Exception as e:
        return f"Unable to query squeue: {e}"


def collect_pipeline_progress():
    """Count completed files across each stage of Study 2."""
    n_ics = len(glob.glob(os.path.join(ROOT_DIR, "make_ICs", "initial_conditions_unbiased", "IC_*.hdf5")))
    n_swift_done = len(glob.glob("/home/cozma/swift_sims/outputs_unbiased/IC_*/snapshot_0100.hdf5"))
    n_labels = len(glob.glob(os.path.join(ROOT_DIR, "GNN-jed", "labels_clean", "IC_*", "B_L_snapshot_0100.npy")))
    n_obs_clean = len(glob.glob(os.path.join(ROOT_DIR, "GNN-jed", "obs_clean", "IC_*", "obs_matrices_snapshot_0100.npy")))
    n_obs_noisy = len(glob.glob(os.path.join(ROOT_DIR, "GNN-jed", "obs_gaia_noisy", "IC_*", "obs_matrices_snapshot_0100.npy")))
    n_graphs_clean = len(glob.glob("/scratch/cozma/graph_data_clean/IC_*/graph_data_output_100_rotation_0.pt"))
    n_graphs_noisy = len(glob.glob("/scratch/cozma/graph_data_noisy/IC_*/graph_data_output_100_rotation_0.pt"))

    logs = sorted(glob.glob(os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "**", "training_metrics_*.log"), recursive=True))
    latest_epochs = []
    for lg in logs:
        rel = os.path.relpath(lg, os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2"))
        try:
            with open(lg, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("===") and not ln.startswith("Epoch") and not ln.startswith("Val") and not ln.startswith("Test")]
            if lines:
                latest_epochs.append(f"{rel}: {lines[-1]}")
        except Exception:
            pass

    return {
        "n_ics": n_ics,
        "n_swift_done": n_swift_done,
        "n_labels": n_labels,
        "n_obs_clean": n_obs_clean,
        "n_obs_noisy": n_obs_noisy,
        "n_graphs_clean": n_graphs_clean,
        "n_graphs_noisy": n_graphs_noisy,
        "latest_epochs": latest_epochs,
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
        lines.append("- Step 5/6 Training Logs:")
        for le in prog["latest_epochs"]:
            lines.append(f"  * {le}")
    lines.append("\nSLURM Queue:")
    lines.append(squeue_str if squeue_str else "(No active jobs)")
    return "\n".join(lines)


def auto_push_to_github():
    """Stage LIVE_STATUS.md and any new Study 2 figures/logs, commit, and push to origin/main."""
    try:
        subprocess.run(
            ["git", "add", "LIVE_STATUS.md", "GNN-jed/trained_models/study2/"],
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
                ["git", "push", "origin", "main"],
                cwd=ROOT_DIR,
                check=False,
                timeout=30,
            )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Publish live Study 2 status updates to ntfy.sh and LIVE_STATUS.md.")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="ntfy.sh topic name.")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds (default: 300s = 5 min).")
    parser.add_argument("--once", action="store_true", help="Send a single status snapshot and exit.")
    args = parser.parse_args()

    last_sent_report = None
    uploaded_plots = set()

    while True:
        squeue_str = get_slurm_queue()
        prog = collect_pipeline_progress()
        report = format_status_report(squeue_str, prog)

        status_md_path = os.path.join(ROOT_DIR, "LIVE_STATUS.md")
        with open(status_md_path, "w", encoding="utf-8") as f:
            f.write(f"# Live Pipeline Status on Jed\n\n```text\n{report}\n```\n")

        # Only push notification if state changed or --once
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

        # Automatically push any newly generated PNG or HTML dashboard files
        new_files = glob.glob(
            os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "**", "*.*"),
            recursive=True,
        )
        for fpath in sorted(new_files):
            if fpath.endswith((".png", ".html")):
                mtime = os.path.getmtime(fpath)
                key = (fpath, int(mtime))
                if key not in uploaded_plots:
                    send_ntfy_message(
                        args.topic,
                        f"Updated figure: {os.path.basename(fpath)}",
                        title="EPFL Jed - New Study 2 Figure",
                        attachment_path=fpath,
                    )
                    uploaded_plots.add(key)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
