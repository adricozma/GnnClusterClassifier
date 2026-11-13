"""Automatic selector for the best-performing 4-observable configuration from Study 2 Step 6.

Parses Final Test metrics across all five Step 6 noisy 4-observable ablation runs
(`4obs_no_RA`, `4obs_no_Dec`, `4obs_no_Parallax`, `4obs_no_pmRA`, `4obs_no_pmDec`),
ranks them primarily by test F1-score, balanced accuracy, negative CoreFNR, and
negative test loss, and emits machine-readable `.json` and `.env` configuration files
for Study 2 V2 automated SLURM chaining.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gaia_v2 import OBS_NAMES_V2, OBSERVABLE_SUBSETS


ABLATION_TAGS = (
    "4obs_no_RA",
    "4obs_no_Dec",
    "4obs_no_Parallax",
    "4obs_no_pmRA",
    "4obs_no_pmDec",
)

FINAL_TEST_RE = re.compile(
    r"Final Test Loss\s*:\s*([0-9.eE+-]+)\s*\|\s*"
    r"Acc:\s*([0-9.]+)%\s*\|\s*"
    r"BalAcc:\s*([0-9.]+)%\s*\|\s*"
    r"F1:\s*([0-9.]+)\s*\|\s*"
    r"Prec:\s*([0-9.]+)\s*\|\s*"
    r"Rec:\s*([0-9.]+)"
    r"(?:\s*\|\s*CoreFNR:\s*([0-9.]+)%\s*\|\s*CoreRec:\s*([0-9.]+)%)?"
)

VAL_EPOCH_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.]+)%\s*\|\s*"
    r"([0-9.eE+-]+)\s*\|\s*([0-9.]+)%\s*\|.*?"
    r"(?:CoreFNR:\s*([0-9.]+)%)?",
    re.MULTILINE,
)


def parse_training_log(log_path: str) -> Optional[Dict[str, Any]]:
    """Parse Final Test metrics (and best validation metrics) from a Step 6 training log."""
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    test_matches = FINAL_TEST_RE.findall(text)
    val_matches = VAL_EPOCH_RE.findall(text)

    result: Dict[str, Any] = {"log_path": log_path}
    if test_matches:
        last = test_matches[-1]
        result.update(
            {
                "status": "complete",
                "metric_source": "test",
                "test_loss": float(last[0]),
                "test_acc": float(last[1]),
                "test_bal_acc": float(last[2]),
                "test_f1": float(last[3]),
                "test_prec": float(last[4]),
                "test_rec": float(last[5]),
                "test_core_fnr": float(last[6]) if last[6] else 0.0,
                "test_core_rec": float(last[7]) if last[7] else 0.0,
            }
        )
    elif val_matches:
        best_val = min(val_matches, key=lambda m: float(m[3]))
        result.update(
            {
                "status": "incomplete",
                "metric_source": "val_best",
                "val_epoch": int(best_val[0]),
                "val_loss": float(best_val[3]),
                "val_acc": float(best_val[4]),
                "val_core_fnr": float(best_val[5]) if best_val[5] else 0.0,
            }
        )
    else:
        return None

    return result


def select_best_4d_config(
    study2_models_dir: str,
    require_all: bool = True,
) -> Dict[str, Any]:
    """Evaluate all five Step 6 noisy 4D ablation runs and select the best configuration."""
    candidates: List[Dict[str, Any]] = []
    missing_or_incomplete: List[str] = []

    for tag in ABLATION_TAGS:
        noisy_log = os.path.join(
            study2_models_dir, f"{tag}_noisy", f"training_metrics_gat_{tag}_noisy.log"
        )
        clean_log = os.path.join(
            study2_models_dir, f"{tag}_clean", f"training_metrics_gat_{tag}_clean.log"
        )
        parsed_noisy = parse_training_log(noisy_log)
        parsed_clean = parse_training_log(clean_log)

        if parsed_noisy is None or (require_all and parsed_noisy.get("status") != "complete"):
            missing_or_incomplete.append(f"{tag} ({noisy_log})")
            if parsed_noisy is None:
                continue

        feat_indices = list(OBSERVABLE_SUBSETS[tag])
        dropped_idx = [i for i in range(5) if i not in feat_indices][0]
        entry: Dict[str, Any] = {
            "tag": tag,
            "dropped_observable": OBS_NAMES_V2[dropped_idx],
            "kept_observables": [OBS_NAMES_V2[i] for i in feat_indices],
            "feature_indices": feat_indices,
            "feature_indices_str": ",".join(str(i) for i in feat_indices),
            "noisy": parsed_noisy,
            "clean": parsed_clean,
        }
        if parsed_noisy["metric_source"] == "test":
            entry["rank_key"] = (
                parsed_noisy["test_f1"],
                parsed_noisy["test_bal_acc"],
                -parsed_noisy["test_core_fnr"],
                -parsed_noisy["test_loss"],
                parsed_noisy["test_acc"],
            )
        else:
            entry["rank_key"] = (
                0.0,
                parsed_noisy["val_acc"],
                -parsed_noisy["val_core_fnr"],
                -parsed_noisy["val_loss"],
                0.0,
            )
        candidates.append(entry)

    if require_all and missing_or_incomplete:
        raise RuntimeError(
            "Incomplete or missing Step 6 noisy 4D logs required for V2 selection: "
            + ", ".join(missing_or_incomplete)
        )
    if not candidates:
        raise RuntimeError(
            f"No valid Step 6 4D ablation logs found inside '{study2_models_dir}'."
        )

    ranked = sorted(candidates, key=lambda c: c["rank_key"], reverse=True)
    winner = ranked[0]
    for c in ranked:
        c["rank_key"] = list(c["rank_key"])

    return {
        "best_tag": winner["tag"],
        "dropped_observable": winner["dropped_observable"],
        "kept_observables": winner["kept_observables"],
        "feature_indices": winner["feature_indices"],
        "feature_indices_str": winner["feature_indices_str"],
        "winner_metrics": winner["noisy"],
        "ranking": ranked,
    }


def write_selection_outputs(selection: Dict[str, Any], out_dir: str) -> None:
    """Persist selection results as JSON and bash-sourceable `.env` files."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "best_4d_config.json")
    env_path = os.path.join(out_dir, "best_4d_config.env")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    kept_csv = ",".join(selection["kept_observables"])
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(f"BEST_4D_TAG={selection['best_tag']}\n")
        f.write(f"BEST_4D_DROPPED={selection['dropped_observable']}\n")
        f.write(f"BEST_4D_KEPT=\"{kept_csv}\"\n")
        f.write(f"BEST_4D_FEATURE_INDICES=\"{selection['feature_indices_str']}\"\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best Step 6 4-observable configuration for Study 2 V2."
    )
    parser.add_argument(
        "--study2_models_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2"),
        help="Directory holding Step 6 4obs_no_*_noisy/ subdirectories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "GNN-jed", "trained_models", "study2_v2"),
        help="Output directory for best_4d_config.json and best_4d_config.env.",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow selection from incomplete/partial Step 6 logs (for dry tests).",
    )
    args = parser.parse_args()

    selection = select_best_4d_config(
        args.study2_models_dir, require_all=not args.allow_partial
    )
    write_selection_outputs(selection, args.out_dir)

    print("=== Study 2 Step 6 Noisy 4D Ranking ===")
    for idx, item in enumerate(selection["ranking"], start=1):
        m = item["noisy"]
        if m["metric_source"] == "test":
            print(
                f"#{idx} {item['tag']:18s} | F1={m['test_f1']:.4f} | "
                f"BalAcc={m['test_bal_acc']:.2f}% | CoreFNR={m['test_core_fnr']:.2f}% | "
                f"Loss={m['test_loss']:.4f} | idx={item['feature_indices_str']}"
            )
        else:
            print(
                f"#{idx} {item['tag']:18s} | ValLoss={m['val_loss']:.4f} | "
                f"ValAcc={m['val_acc']:.2f}% | idx={item['feature_indices_str']} (partial)"
            )
    print(
        f"\nSelected Best 4D configuration: {selection['best_tag']} "
        f"(drops {selection['dropped_observable']}, keeps {selection['kept_observables']}, "
        f"indices={selection['feature_indices_str']})"
    )


if __name__ == "__main__":
    main()
