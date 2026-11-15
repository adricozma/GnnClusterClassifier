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
import re
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


def get_in_flight_training_progress():
    """Detect active batch progress (including 4D ablations and V2 runs) from the most recent SLURM output log."""
    out_files = sorted(
        glob.glob(os.path.join(ROOT_DIR, "slurm_logs", "*.out")),
        key=os.path.getmtime,
        reverse=True,
    )
    for out_f in out_files[:4]:
        # Only consider logs updated within the last 90 minutes
        try:
            if (time.time() - os.path.getmtime(out_f)) > 5400:
                continue
            with open(out_f, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-160:]
            exp_tag = "clean_5obs"
            arch_lbl = "GAT"
            for ln in lines:
                m_idx = re.search(r"graph_data_([A-Za-z0-9_]+)", ln)
                if m_idx:
                    exp_tag = m_idx.group(1)
                m_save = re.search(r"study2/([A-Za-z0-9_]+)/", ln)
                if m_save and m_save.group(1) != "weights":
                    exp_tag = m_save.group(1)
                if "train_gcn.py" in ln or "[GCN]" in ln:
                    arch_lbl = "GCN"
                elif "train_gat.py" in ln or "[GAT]" in ln:
                    arch_lbl = "GAT"

            for line in reversed(lines):
                m = re.search(
                    r"Epoch\s+(\d+)\s*\|\s*Batch\s+(\d+)/(\d+)\s*\|\s*Batch Loss:\s*([0-9.]+)\s*\|\s*LR:\s*([0-9.eE+-]+)\s*\|\s*RAM:\s*([0-9.]+[GM])",
                    line,
                )
                if m:
                    ep = int(m.group(1))
                    b_cur = int(m.group(2))
                    b_tot = int(m.group(3))
                    pct = (b_cur / max(1, b_tot)) * 100.0
                    loss = float(m.group(4))
                    lr = m.group(5)
                    ram = m.group(6)
                    return {
                        "summary": f"[IN FLIGHT] {exp_tag} ({arch_lbl}): Ep {ep:03d} | Batch {b_cur:04d}/{b_tot:04d} ({pct:.1f}%) | Batch Loss {loss:.4f} | LR: {lr} | RAM: {ram}",
                        "row": (exp_tag, f"{arch_lbl} (In Flight)", f"Ep {ep:02d} ({pct:.0f}%)", loss, "-", "-", "-", "-", "-", "In progress", "-"),
                    }
        except Exception:
            pass
    return None


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
        if rel.startswith("legacy"):
            continue
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

            core_fnr_str = "-"
            if d.get("test_core_fnr") is not None:
                core_fnr_str = f"{d['test_core_fnr']:.2f}%"
                if d.get("val_core_fnr"):
                    core_fnr_str += f" (val: {min(d['val_core_fnr']):.2f}%)"
            elif d.get("val_core_fnr"):
                core_fnr_str = f"{min(d['val_core_fnr']):.2f}% (val min)"

            core_ep_str = ""
            if d.get("val_core_fnr"):
                core_ep_str = f" | CoreFNR: {d['val_core_fnr'][-1]:.2f}% (min={min(d['val_core_fnr']):.2f}%)"

            latest_epochs.append(
                f"{rel}: Ep {ep:03d} | Train {tr_l:.4f} ({tr_a:.2f}%) | Val {va_l:.4f} ({va_a:.2f}%, Best={best_va:.2f}%){core_ep_str} | Test={test_str}"
            )
            exp_name = os.path.basename(os.path.dirname(lg))
            model_name = "GAT" if "_gat" in os.path.basename(lg) else "GCN"
            training_rows.append((exp_name, model_name, ep, tr_l, tr_a, va_l, va_a, best_va, core_fnr_str, test_str, f1_str))

    in_flight = get_in_flight_training_progress()
    if in_flight:
        latest_epochs.append(in_flight["summary"])
        training_rows.append(in_flight["row"])

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
        if "/legacy/" in pth_path.replace("\\", "/"):
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
            "| Experiment | Architecture | Latest Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc | Core FNR | Test Acc | Test F1 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in prog["training_rows"]:
            if len(row) == 11:
                exp_name, model_name, ep, tr_l, tr_a, va_l, va_a, best_va, core_str, test_str, f1_str = row
            else:
                exp_name, model_name, ep, tr_l, tr_a, va_l, va_a, best_va, test_str, f1_str = row
                core_str = "-"
            tr_l_s = f"{tr_l:.4f}" if isinstance(tr_l, (int, float)) else str(tr_l)
            tr_a_s = f"{tr_a:.2f}%" if isinstance(tr_a, (int, float)) else str(tr_a)
            va_l_s = f"{va_l:.4f}" if isinstance(va_l, (int, float)) else str(va_l)
            va_a_s = f"{va_a:.2f}%" if isinstance(va_a, (int, float)) else str(va_a)
            best_va_s = f"**{best_va:.2f}%**" if isinstance(best_va, (int, float)) else str(best_va)
            md_lines.append(
                f"| `{exp_name}` | **{model_name}** | {ep} | {tr_l_s} | {tr_a_s} | {va_l_s} | {va_a_s} | {best_va_s} | {core_str} | **{test_str}** | {f1_str} |"
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

    svd_plots = sorted(glob.glob(os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "**", "spectral_analysis_GNN_*.png"), recursive=True))
    if svd_plots:
        md_lines.extend([
            "",
            "## RMT Singular Value Candle Spectra (Marchenko-Pastur Bulk vs Detached Signal)",
            "",
        ])
        for svd_p in svd_plots:
            rel_svd = os.path.relpath(svd_p, ROOT_DIR)
            exp_folder = os.path.basename(os.path.dirname(svd_p))
            md_lines.extend([
                f"### `{exp_folder}`",
                f"![RMT Candle Spectrum {exp_folder}]({rel_svd})",
                "",
            ])

    with open(status_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    try:
        write_master_html_index(report, prog)
    except Exception:
        pass


def _friendly_plot_label(fname: str) -> str:
    """Return a clean, sober French scientific label for a figure filename."""
    base = fname.rsplit(".", 1)[0]
    if base.startswith("spectral_analysis_GNN_"):
        tag = base.replace("spectral_analysis_GNN_", "")
        return f"Spectre RMT (Valeurs Singulières / Bâtons) — {tag}"
    if base.startswith("live_plots_GNN_"):
        tag = base.replace("live_plots_GNN_", "")
        return f"Courbes d'Entraînement (Loss & Accuracy) — {tag}"
    if base.startswith("surviving_core_fnr_by_perigalacticon_"):
        rest = base.replace("surviving_core_fnr_by_perigalacticon_", "")
        return f"Core FNR & Rappel par Péricentre R_p1 — {rest}"
    if base.startswith("error_percentage_by_perigalacticon_"):
        rest = base.replace("error_percentage_by_perigalacticon_", "")
        return f"Erreur de Classification par Péricentre R_p1 — {rest}"
    if base.startswith("prediction_"):
        rest = base.replace("prediction_", "")
        return f"Prédiction 3D Amas & Queues de Marée — {rest}"
    if base == "R_vs_t_6panels_rp1":
        return "Évolution Orbitale R(t) & Fraction Liée (6 Panneaux)"
    if base == "seeds_positions_unbiased":
        return "Distribution Galactocentrique des 81 Orbites (Seeds)"
    return fname


def _normalize_existing_plotly_html_to_white(study2_dir: str) -> None:
    """Ensure existing standalone Plotly HTML files use a clean white background matching index.html."""
    for hpath in glob.glob(os.path.join(study2_dir, "*", "*.html")):
        if os.path.basename(hpath) == "index.html":
            continue
        try:
            with open(hpath, "r", encoding="utf-8") as f:
                txt = f.read()
            if "#0f172a" in txt or "#1e293b" in txt or "plotly_dark" in txt:
                new_txt = (
                    txt.replace("background-color: #0f172a;", "background-color: #ffffff;")
                    .replace("background: #0f172a;", "background: #ffffff;")
                    .replace("background: #1e293b;", "background: #ffffff; border: 1px solid #e5e7eb;")
                    .replace('paper_bgcolor: "#1e293b"', 'template: "plotly_white", paper_bgcolor: "#ffffff"')
                    .replace('plot_bgcolor: "#0f172a"', 'plot_bgcolor: "#ffffff"')
                    .replace('"paper_bgcolor": "#0f172a"', '"paper_bgcolor": "#ffffff"')
                    .replace('"plot_bgcolor": "#1e293b"', '"plot_bgcolor": "#ffffff"')
                    .replace('"template": "plotly_dark"', '"template": "plotly_white"')
                    .replace('color: "#e2e8f0"', 'color: "#111827"')
                    .replace('"color": "#f8fafc"', '"color": "#111827"')
                    .replace("color: #f8fafc;", "color: #111827;")
                    .replace("color: #e2e8f0;", "color: #111827;")
                )
                if new_txt != txt:
                    with open(hpath, "w", encoding="utf-8") as f:
                        f.write(new_txt)
        except Exception:
            pass


def write_master_html_index(report, prog):
    """Write a unified, single-page classified index.html at repo root and in study2/ with a sober white academic style."""
    study2_dir = os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2")
    os.makedirs(study2_dir, exist_ok=True)
    _normalize_existing_plotly_html_to_white(study2_dir)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    expected_ablations = [
        ("4obs_no_RA_clean", "Sans Ascension Droite (α / RA) — Indices [1,2,3,4]"),
        ("4obs_no_DEC_clean", "Sans Déclinaison (δ / Dec) — Indices [0,2,3,4]"),
        ("4obs_no_parallax_clean", "Sans Parallaxe (ϖ) — Indices [0,1,3,4]"),
        ("4obs_no_pmra_clean", "Sans Mouvement Propre μ_α* — Indices [0,1,2,4]"),
        ("4obs_no_pmdec_clean", "Sans Mouvement Propre μ_δ — Indices [0,1,2,3]"),
    ]

    all_subdirs = sorted([
        d for d in os.listdir(study2_dir)
        if os.path.isdir(os.path.join(study2_dir, d)) and d not in ("weights", "legacy")
    ])
    v2_dirs = [d for d in all_subdirs if d.startswith("v2_")]
    ablation_dirs = [d for d in all_subdirs if d.startswith("4obs_")]
    for ab_name, _ in expected_ablations:
        if ab_name not in ablation_dirs:
            ablation_dirs.append(ab_name)

    def _render_for_base(base_prefix: str) -> str:
        def _url(rel_within_study2: str) -> str:
            return f"{base_prefix}{rel_within_study2}"

        rows_html = []
        for row in prog.get("training_rows", []):
            if len(row) == 11:
                exp_name, model_name, ep, tr_l, tr_a, va_l, va_a, best_va, core_str, test_str, f1_str = row
            else:
                exp_name, model_name, ep, tr_l, tr_a, va_l, va_a, best_va, test_str, f1_str = row
                core_str = "-"
            tr_l_s = f"{tr_l:.4f}" if isinstance(tr_l, (int, float)) else str(tr_l)
            tr_a_s = f"{tr_a:.2f}%" if isinstance(tr_a, (int, float)) else str(tr_a)
            va_l_s = f"{va_l:.4f}" if isinstance(va_l, (int, float)) else str(va_l)
            va_a_s = f"{va_a:.2f}%" if isinstance(va_a, (int, float)) else str(va_a)
            best_va_s = f"{best_va:.2f}%" if isinstance(best_va, (int, float)) else str(best_va)

            badge_cls = "badge-gat" if "GAT" in model_name else "badge-gcn"
            if "In Flight" in model_name:
                badge_cls = "badge-live"
            status_cls = "status-live" if test_str == "In progress" else "status-done"
            rows_html.append(
                f"<tr>"
                f"<td><code>{exp_name}</code></td>"
                f"<td><span class='badge {badge_cls}'>{model_name}</span></td>"
                f"<td>{ep}</td>"
                f"<td>{tr_l_s}</td>"
                f"<td>{tr_a_s}</td>"
                f"<td>{va_l_s}</td>"
                f"<td>{va_a_s}</td>"
                f"<td><strong>{best_va_s}</strong></td>"
                f"<td>{core_str}</td>"
                f"<td><span class='{status_cls}'>{test_str}</span></td>"
                f"<td><strong>{f1_str}</strong></td>"
                f"</tr>"
            )
        metrics_table_body = "\n".join(rows_html) if rows_html else "<tr><td colspan='11'>Aucun entraînement trouvé</td></tr>"

        rmt_cards = []
        for exp_d in sorted(all_subdirs):
            svd_png = os.path.join(study2_dir, exp_d, f"spectral_analysis_GNN_{exp_d}.png")
            svd_html = os.path.join(study2_dir, exp_d, f"spectral_analysis_GNN_{exp_d}.html")
            live_png = os.path.join(study2_dir, exp_d, f"live_plots_GNN_{exp_d}.png")
            live_html = os.path.join(study2_dir, exp_d, f"live_plots_GNN_{exp_d}.html")
            if os.path.exists(svd_png) or os.path.exists(live_png):
                btns = []
                if os.path.exists(svd_html):
                    u_svd = _url(f"{exp_d}/spectral_analysis_GNN_{exp_d}.html")
                    btns.append(f"<button class='btn-plot' onclick=\"openInViewer('{u_svd}', 'Spectre RMT — {exp_d}')\">Spectre RMT Interactif</button>")
                    btns.append(f"<a class='btn-ext' href='{u_svd}' target='_blank'>Nouvel onglet ↗</a>")
                if os.path.exists(live_html):
                    u_live = _url(f"{exp_d}/live_plots_GNN_{exp_d}.html")
                    btns.append(f"<button class='btn-plot' onclick=\"openInViewer('{u_live}', 'Courbes Loss/Acc — {exp_d}')\">Courbes Interactives</button>")
                imgs_html = ""
                if os.path.exists(svd_png):
                    u_sp = _url(f"{exp_d}/spectral_analysis_GNN_{exp_d}.png")
                    imgs_html += f"<div class='img-wrap'><h4>Spectre des Valeurs Singulières en Bâtons (Candles) vs Bulk Marchenko–Pastur</h4><a href='{u_sp}' target='_blank'><img src='{u_sp}' alt='SVD {exp_d}' /></a></div>"
                if os.path.exists(live_png):
                    u_lp = _url(f"{exp_d}/live_plots_GNN_{exp_d}.png")
                    imgs_html += f"<div class='img-wrap'><h4>Évolution Loss & Accuracy</h4><a href='{u_lp}' target='_blank'><img src='{u_lp}' alt='Live {exp_d}' /></a></div>"
                rmt_cards.append(
                    f"<div class='exp-block'>"
                    f"<div class='exp-header'><h3>Expérience : <code>{exp_d}</code></h3><div class='btn-row'>{''.join(btns)}</div></div>"
                    f"<div class='dual-grid'>{imgs_html}</div>"
                    f"</div>"
                )
        rmt_overview_html = "\n".join(rmt_cards) if rmt_cards else "<p>Aucun spectre RMT généré pour le moment.</p>"

        def _build_experiment_section(exp_d: str, title: str, subtitle: str) -> str:
            exp_path = os.path.join(study2_dir, exp_d)
            if not os.path.isdir(exp_path):
                return (
                    f"<div class='exp-block pending-block'>"
                    f"<h3>{title} (<code>{exp_d}</code>)</h3>"
                    f"<p class='subtitle'>{subtitle} — <strong>En file d'attente SLURM (Pending)</strong></p>"
                    f"</div>"
                )
            html_files = sorted([
                f for f in os.listdir(exp_path)
                if f.endswith(".html") and not f.startswith("study2_interactive_dashboard")
            ])
            def _sort_key(fn: str):
                if fn.startswith("spectral_analysis"):
                    return (0, fn)
                if fn.startswith("live_plots"):
                    return (1, fn)
                if fn.startswith("surviving_core"):
                    return (2, fn)
                if fn.startswith("error_percentage"):
                    return (3, fn)
                if fn.startswith("R_vs_t"):
                    return (4, fn)
                if fn.startswith("seeds_"):
                    return (5, fn)
                return (6, fn)

            html_files.sort(key=_sort_key)
            png_files = sorted([f for f in os.listdir(exp_path) if f.endswith(".png")], key=_sort_key)

            btn_items = []
            for hf in html_files:
                u_hf = _url(f"{exp_d}/{hf}")
                lbl = _friendly_plot_label(hf)
                btn_items.append(
                    f"<div class='plot-pill'>"
                    f"<button class='btn-plot' onclick=\"openInViewer('{u_hf}', '{lbl}')\">{lbl}</button>"
                    f"<a class='btn-ext' href='{u_hf}' target='_blank' title='Ouvrir dans un nouvel onglet'>↗</a>"
                    f"</div>"
                )
            btns_block = (
                f"<div class='pills-container'>{''.join(btn_items)}</div>"
                if btn_items
                else "<p class='muted'>Aucun graphe HTML interactif disponible pour le moment.</p>"
            )

            png_cards = []
            for pf in png_files:
                u_pf = _url(f"{exp_d}/{pf}")
                lbl = _friendly_plot_label(pf)
                matching_html = pf.rsplit(".", 1)[0] + ".html"
                open_interactive_btn = ""
                if matching_html in html_files:
                    u_mh = _url(f"{exp_d}/{matching_html}")
                    open_interactive_btn = f"<button class='btn-card-action' onclick=\"openInViewer('{u_mh}', '{lbl}')\">Vue Interactive Plotly</button>"
                png_cards.append(
                    f"<div class='card'>"
                    f"<div class='card-top'><h4>{lbl}</h4>{open_interactive_btn}</div>"
                    f"<p class='filename'><code>{exp_d}/{pf}</code></p>"
                    f"<a href='{u_pf}' target='_blank'><img src='{u_pf}' alt='{pf}' loading='lazy' /></a>"
                    f"</div>"
                )
            png_grid = (
                f"<div class='grid'>{''.join(png_cards)}</div>"
                if png_cards
                else "<p class='muted'>Figures PNG en cours de génération...</p>"
            )

            return (
                f"<div class='exp-block'>"
                f"<h3>{title} — <code>{exp_d}</code></h3>"
                f"<p class='subtitle'>{subtitle}</p>"
                f"<h4>Graphes Interactifs Plotly (Cliquer pour afficher dans le lecteur intégré ou ↗ pour nouvel onglet)</h4>"
                f"{btns_block}"
                f"<h4>Figures Statiques Haute Résolution (PNG)</h4>"
                f"{png_grid}"
                f"</div>"
            )

        clean_section_html = _build_experiment_section(
            "clean_5obs",
            "1. Baseline 5D Sans Bruit (Observables Exactes)",
            "Comparaison directe GAT (3 couches, 4 têtes, résiduel, Test Acc = 93.04%, Core FNR = 6.49%) vs GCN (Baseline isotrope, Test Acc = 92.17%).",
        )
        noisy_section_html = _build_experiment_section(
            "noisy_5obs",
            "2. Modèle 5D Bruité Gaia DR3 (Curriculum Fine-Tuning)",
            "Évaluation de la robustesse face aux incertitudes astrométriques dépendantes de la magnitude G (Gaia DR3).",
        )

        ablation_blocks = []
        for ab_dir, ab_desc in expected_ablations:
            ablation_blocks.append(
                _build_experiment_section(
                    ab_dir,
                    f"Ablation 4D : {ab_dir}",
                    f"Leave-One-Observable-Out ({ab_desc}) — Transfert 5D→4D depuis best_gat_clean.pth (93.04%).",
                )
            )
        ablations_section_html = "\n".join(ablation_blocks)

        v2_blocks = []
        if v2_dirs:
            for v2_d in v2_dirs:
                v2_blocks.append(
                    _build_experiment_section(
                        v2_d,
                        f"Study 2 V2 : {v2_d}",
                        "Adaptation au domaine réel Gaia DR3 (Matrice de covariance 5×5 + Normalisation d'échelle physique + Pondération 3 classes).",
                    )
                )
        else:
            v2_blocks.append(
                "<div class='exp-block pending-block'>"
                "<h3>Study 2 V2 — Domaine Réel Gaia DR3 (En file d'attente SLURM : Jobs 41185009–41185012)</h3>"
                "<p class='subtitle'>Cette section se remplira automatiquement dès le démarrage des jobs V2 (Génération covariances Gaia DR3, Graphes à échelle physique préservée, GAT/GCN 3 classes : Bound Core / Potential Escaper / Unbound).</p>"
                "</div>"
            )
        v2_section_html = "\n".join(v2_blocks)

        weights_rows = []
        for fname, rel_pth, rel_central, size_kb in prog.get("saved_weights", []):
            weights_rows.append(
                f"<tr><td><code>{fname}</code></td><td>{size_kb:.1f} KB</td><td><code>{rel_pth}</code></td><td><code>{rel_central}</code></td></tr>"
            )
        weights_table_html = "\n".join(weights_rows) if weights_rows else "<tr><td colspan='4'>Aucun fichier .pth</td></tr>"

        master_live_url = _url("latest_live_training.png")
        default_iframe_url = _url("clean_5obs/spectral_analysis_GNN_clean_5obs.html")

        return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Study 2 — Index des Entraînements & Figures GNN</title>
  <style>
    :root {{
      --bg: #ffffff;
      --panel: #ffffff;
      --panel-alt: #f9fafb;
      --border: #e5e7eb;
      --border-strong: #d1d5db;
      --text: #111827;
      --muted: #4b5563;
      --accent: #1d4ed8;
      --accent-soft: #eff6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 0;
      line-height: 1.5;
    }}
    header {{
      background: #f9fafb;
      border-bottom: 1px solid var(--border);
      padding: 18px 28px;
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    h1 {{ margin: 0; font-size: 1.25rem; font-weight: 600; color: #111827; }}
    h2 {{ font-size: 1.15rem; font-weight: 600; color: #111827; margin-top: 24px; margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
    h3 {{ font-size: 1.02rem; font-weight: 600; color: #111827; margin: 0; }}
    h4 {{ font-size: 0.9rem; font-weight: 600; color: #374151; margin: 14px 0 8px 0; }}
    .timestamp {{ font-size: 0.82rem; color: var(--muted); background: #ffffff; padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); }}
    .nav-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 12px auto 0;
      max-width: 1600px;
    }}
    .tab-btn {{
      background: #ffffff;
      color: #374151;
      border: 1px solid var(--border-strong);
      padding: 7px 14px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: 500;
      font-size: 0.86rem;
    }}
    .tab-btn:hover {{ background: #f3f4f6; color: #111827; }}
    .tab-btn.active {{
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
    }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 20px 28px 60px; }}
    .tab-section {{ display: none; }}
    .tab-section.active {{ display: block; }}
    .viewer-box {{
      background: #ffffff;
      border: 1px solid var(--border-strong);
      border-radius: 6px;
      padding: 12px 14px;
      margin-bottom: 24px;
    }}
    .viewer-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .viewer-title {{ font-weight: 600; color: #111827; font-size: 0.94rem; }}
    iframe#plotly-viewer {{
      width: 100%;
      height: 520px;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: #ffffff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid var(--border);
      margin-bottom: 24px;
      font-size: 0.88rem;
    }}
    th, td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid var(--border); }}
    th {{ background: #f9fafb; color: #374151; font-weight: 600; font-size: 0.8rem; }}
    td:first-child, th:first-child {{ text-align: left; }}
    tr:hover td {{ background: #f9fafb; }}
    code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 3px; font-size: 0.84em; color: #1f2937; }}
    .badge {{ padding: 2px 7px; border-radius: 4px; font-weight: 600; font-size: 0.78rem; }}
    .badge-gat {{ background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }}
    .badge-gcn {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }}
    .badge-live {{ background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }}
    .status-done {{ color: #15803d; font-weight: 600; }}
    .status-live {{ color: #b45309; font-weight: 600; }}
    .exp-block {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 18px;
      margin-bottom: 20px;
    }}
    .pending-block {{ background: #f9fafb; border-style: dashed; }}
    .exp-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      margin-bottom: 12px;
    }}
    .btn-row {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }}
    .subtitle {{ color: var(--muted); margin-top: 4px; margin-bottom: 14px; font-size: 0.9rem; }}
    .pills-container {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 16px;
    }}
    .plot-pill {{
      display: inline-flex;
      align-items: center;
      background: #f9fafb;
      border: 1px solid var(--border-strong);
      border-radius: 4px;
      overflow: hidden;
    }}
    .btn-plot {{
      background: transparent;
      color: #1f2937;
      border: none;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 0.83rem;
      font-weight: 500;
    }}
    .btn-plot:hover {{ background: var(--accent-soft); color: var(--accent); }}
    .btn-ext {{
      color: var(--accent);
      padding: 6px 8px;
      text-decoration: none;
      border-left: 1px solid var(--border);
      font-size: 0.8rem;
      background: #ffffff;
    }}
    .btn-ext:hover {{ background: var(--accent-soft); }}
    .dual-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
      gap: 16px;
    }}
    .img-wrap {{
      background: #ffffff;
      padding: 10px;
      border-radius: 4px;
      border: 1px solid var(--border);
    }}
    .img-wrap h4 {{ margin: 0 0 8px 0; font-size: 0.86rem; color: #374151; }}
    .img-wrap img {{ width: 100%; height: auto; display: block; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 6px;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }}
    .card-top h4 {{ margin: 0; font-size: 0.9rem; color: #111827; }}
    .btn-card-action {{
      background: #ffffff;
      color: var(--accent);
      border: 1px solid var(--border-strong);
      border-radius: 4px;
      padding: 4px 9px;
      cursor: pointer;
      font-size: 0.78rem;
      font-weight: 500;
      white-space: nowrap;
    }}
    .btn-card-action:hover {{ background: var(--accent-soft); border-color: var(--accent); }}
    .filename {{ margin: 0 0 8px 0; font-size: 0.76rem; color: var(--muted); }}
    .card img {{ width: 100%; height: auto; display: block; border-top: 1px solid #f3f4f6; padding-top: 6px; }}
    pre.status-box {{
      background: #f9fafb;
      color: #1f2937;
      padding: 14px;
      border-radius: 4px;
      border: 1px solid var(--border);
      overflow-x: auto;
      font-size: 0.83rem;
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-top">
      <h1>Study 2 — Index des Entraînements & Analyses GNN (GAT vs GCN)</h1>
      <span class="timestamp">Dernière mise à jour : {now_str}</span>
    </div>
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('tab-overview', this)">Vue Globale & Spectres RMT</button>
      <button class="tab-btn" onclick="switchTab('tab-clean', this)">1. Baseline 5D Sans Bruit (clean_5obs)</button>
      <button class="tab-btn" onclick="switchTab('tab-noisy', this)">2. Gaia DR3 5D Bruité (noisy_5obs)</button>
      <button class="tab-btn" onclick="switchTab('tab-ablations', this)">3. Ablations 4D (4obs_*)</button>
      <button class="tab-btn" onclick="switchTab('tab-v2', this)">4. Study 2 V2 (Domaine Gaia Réel)</button>
      <button class="tab-btn" onclick="switchTab('tab-weights', this)">5. Poids (.pth) & Queue SLURM</button>
    </div>
  </header>

  <main>
    <div class="viewer-box" id="viewer-container">
      <div class="viewer-bar">
        <span class="viewer-title" id="viewer-label">Lecteur Interactif Plotly : Spectre RMT à Bâtons (clean_5obs)</span>
        <div class="btn-row">
          <a id="viewer-ext-link" class="btn-card-action" href="{default_iframe_url}" target="_blank" style="text-decoration:none;">Ouvrir dans un nouvel onglet ↗</a>
          <button class="btn-card-action" onclick="toggleViewer()">Masquer / Afficher le lecteur</button>
        </div>
      </div>
      <iframe id="plotly-viewer" src="{default_iframe_url}"></iframe>
    </div>

    <!-- TAB 1: VUE GLOBALE & SPECTRES RMT -->
    <section id="tab-overview" class="tab-section active">
      <h2>Tableau Récapitulatif des Entraînements</h2>
      <table>
        <thead>
          <tr>
            <th>Expérience</th>
            <th>Architecture</th>
            <th>Époque</th>
            <th>Train Loss</th>
            <th>Train Acc</th>
            <th>Val Loss</th>
            <th>Val Acc</th>
            <th>Best Val Acc</th>
            <th>Core FNR</th>
            <th>Test Acc</th>
            <th>Test F1</th>
          </tr>
        </thead>
        <tbody>
          {metrics_table_body}
        </tbody>
      </table>

      <div class="exp-block">
        <h3>Courbes d'Apprentissage Globales (Tous Modèles Superposés)</h3>
        <a href="{master_live_url}" target="_blank"><img src="{master_live_url}" alt="Master Live Training Curves" style="width:100%; max-height:520px; object-fit:contain; margin-top:10px;" /></a>
      </div>

      <h2>Comparaison des Spectres des Valeurs Singulières en Bâtons (Candles RMT vs Bulk Marchenko–Pastur)</h2>
      <p class="subtitle">À gauche : densité spectrale empirique avec le bulk de bruit Marchenko–Pastur (zone ombrée) et les valeurs singulières qui s'en détachent (tiges verticales). À droite : spectre complet classé par rang.</p>
      {rmt_overview_html}
    </section>

    <!-- TAB 2: CLEAN 5OBS -->
    <section id="tab-clean" class="tab-section">
      {clean_section_html}
    </section>

    <!-- TAB 3: NOISY 5OBS -->
    <section id="tab-noisy" class="tab-section">
      {noisy_section_html}
    </section>

    <!-- TAB 4: ABLATIONS 4D -->
    <section id="tab-ablations" class="tab-section">
      <h2>Études d'Ablation 4D (Leave-One-Observable-Out)</h2>
      <p class="subtitle">Évaluation de l'importance physique de chaque observable astrométrique en retirant tour à tour α, δ, ϖ, μ_α*, μ_δ.</p>
      {ablations_section_html}
    </section>

    <!-- TAB 5: STUDY 2 V2 -->
    <section id="tab-v2" class="tab-section">
      {v2_section_html}
    </section>

    <!-- TAB 6: WEIGHTS & SLURM -->
    <section id="tab-weights" class="tab-section">
      <h2>Registre des Poids Sauvegardés (.pth Checkpoints)</h2>
      <table>
        <thead>
          <tr><th>Checkpoint</th><th>Taille</th><th>Dossier Expérience</th><th>Archive Centrale</th></tr>
        </thead>
        <tbody>
          {weights_table_html}
        </tbody>
      </table>
      <h2>État du Pipeline & Queue SLURM sur Jed</h2>
      <pre class="status-box">{report}</pre>
    </section>
  </main>

  <script>
    function switchTab(tabId, btnEl) {{
      document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      const target = document.getElementById(tabId);
      if (target) target.classList.add('active');
      if (btnEl) btnEl.classList.add('active');
    }}
    function openInViewer(url, label) {{
      const viewer = document.getElementById('plotly-viewer');
      const lbl = document.getElementById('viewer-label');
      const ext = document.getElementById('viewer-ext-link');
      viewer.style.display = 'block';
      viewer.src = url;
      lbl.textContent = 'Lecteur Interactif Plotly : ' + label;
      ext.href = url;
      document.getElementById('viewer-container').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
    function toggleViewer() {{
      const viewer = document.getElementById('plotly-viewer');
      viewer.style.display = (viewer.style.display === 'none') ? 'block' : 'none';
    }}
  </script>
</body>
</html>
"""

    root_index_path = os.path.join(ROOT_DIR, "index.html")
    study2_index_path = os.path.join(study2_dir, "index.html")
    with open(root_index_path, "w", encoding="utf-8") as f:
        f.write(_render_for_base("GNN-jed/trained_models/study2/"))
    with open(study2_index_path, "w", encoding="utf-8") as f:
        f.write(_render_for_base(""))


def auto_push_to_github():
    """Stage LIVE_STATUS.md, index.html, and any new Study 2 figures/logs/weights, commit, pull --rebase, and push to origin/main."""
    try:
        subprocess.run(
            [
                "git",
                "add",
                "-A",
                ".gitignore",
                ".gitattributes",
                "index.html",
                "README.md",
                "LIVE_STATUS.md",
                "REPO_MAP.md",
                "src/",
                "scripts/",
                "slurm/",
                "tests/",
                "assets/figures/",
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


def daemonize(log_path=None):
    """Detach process via POSIX double-fork and setsid so it survives VS Code/SSH exit."""
    if log_path is None:
        log_path = os.path.join(ROOT_DIR, "slurm_logs", "live_monitor_daemon.log")
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
    watched_scripts = [
        os.path.abspath(__file__),
        os.path.join(ROOT_DIR, "scripts", "live_plot_utils.py"),
        os.path.join(ROOT_DIR, "scripts", "generate_study2_figures.py"),
    ]
    initial_mtimes = {p: os.path.getmtime(p) for p in watched_scripts if os.path.exists(p)}

    while True:
        # Auto-reload daemon via os.execv if any monitoring/plotting script was updated on disk
        if not args.once:
            for p, old_mt in list(initial_mtimes.items()):
                if os.path.exists(p) and os.path.getmtime(p) > old_mt:
                    os.execv(sys.executable, [sys.executable] + sys.argv)
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
            last_sent_report = state_signature
        auto_push_to_github()

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

                # Hot-reload transition for active 4obs_no_RA_clean if it started before the dropout=0.25 & feature_indices=1,2,3,4 fix
                ablation0_marker = os.path.join(
                    ROOT_DIR, "GNN-jed", "trained_models", "study2", "4obs_no_RA_clean", ".ablation0_ep18_reload_done"
                )
                ablation0_log = os.path.join(
                    ROOT_DIR, "GNN-jed", "trained_models", "study2", "4obs_no_RA_clean", "training_metrics_gat_4obs_no_RA_clean.log"
                )
                if os.path.exists(cmdline_file) and not os.path.exists(ablation0_marker):
                    with open(cmdline_file, "rb") as cf:
                        cmd_str = cf.read().decode("utf-8", errors="ignore")
                    if "train_gat.py" in cmd_str and "4obs_no_RA_clean" in cmd_str and os.path.exists(ablation0_log):
                        with open(ablation0_log, "r", encoding="utf-8", errors="ignore") as lf:
                            log_head = lf.read(512)
                        if "dropout=0.4" in log_head:
                            with open(ablation0_marker, "w", encoding="utf-8") as mf:
                                mf.write("triggered\n")
                            for stale_f in (
                                os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "4obs_no_RA_clean", "last_gat_4obs_no_RA_clean.pth"),
                                os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2", "4obs_no_RA_clean", "best_gat_4obs_no_RA_clean.pth"),
                                ablation0_log,
                            ):
                                try:
                                    if os.path.exists(stale_f):
                                        os.remove(stale_f)
                                except OSError:
                                    pass
                            time.sleep(2.0)
                            os.kill(ppid, signal.SIGTERM)
                # Fast-reload transition for active 4obs_no_RA_clean if stuck in micro_bs=8 / force_disk_offload after Epoch 13
                fast_reload_marker = os.path.join(
                    ROOT_DIR, "GNN-jed", "trained_models", "study2", "4obs_no_RA_clean", ".ablation0_ep13_fast_reload_done"
                )
                resume_pt = os.path.join(
                    ROOT_DIR, "GNN-jed", "trained_models", "study2", "4obs_no_RA_clean", "best_gat_4obs_no_RA_clean_resume_state.pt"
                )
                if os.path.exists(cmdline_file) and not os.path.exists(fast_reload_marker):
                    with open(cmdline_file, "rb") as cf:
                        cmd_str = cf.read().decode("utf-8", errors="ignore").replace("\x00", " ")
                    if "train_gat.py" in cmd_str and "4obs_no_RA_clean" in cmd_str and ("--force_disk_offload" in cmd_str or "--micro_bs 8" in cmd_str):
                        if os.path.exists(resume_pt):
                            with open(fast_reload_marker, "w", encoding="utf-8") as mf:
                                mf.write("triggered_fast_reload\n")
                            time.sleep(2.0)
                            os.kill(ppid, signal.SIGTERM)
            except Exception:
                pass
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
