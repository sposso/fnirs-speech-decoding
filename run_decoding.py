"""
Run the full per-subject decoding pipeline (SVC + LDA) for the paper.

Outputs:
  outputs/auc/auc_per_subject.csv
      One row per (subject, model, fold). Long-format AUC table that
      Table I is built from (notebooks/01_decoding_table.ipynb).
  outputs/shap/subject_{i}/shap_values_{model}_subject_{i}.npz
      One npz per (subject, model). Each contains 5 arrays
      ('folder_1' .. 'folder_5'), one per outer CV fold, with shape
      (n_test, n_channels * n_times). Consumed by the figure scripts
      (fig_1_and_3.py, fig_2.py, table_2.py).

Run:
    python run_decoding.py             # all 8 subjects, both models
    python run_decoding.py --subjects 0 1 --models svc
"""
import argparse
import os

import numpy as np
import pandas as pd

from audio_data import final_audio_data
from train_shap import machine_learning_exp


SUBJECTS = list(range(8))
MODELS = ("svc", "lda")
T_MIN, T_MAX = 0.0, 18.0


def run_subject(subject, models, out_root):
    """Train on one subject and dump SHAP + AUCs to disk."""
    print(f"\n=== Subject {subject} ===")
    X_arr, Y, _ = final_audio_data(subject, T_MIN, T_MAX, chroma=None)
    print(f"  X shape: {X_arr.shape}  (trials, channels, time)")

    # Flatten (channels, time) into one feature dimension for SHAP.
    X_flat = X_arr.reshape(X_arr.shape[0], -1)
    X = pd.DataFrame(X_flat)
    X.columns = X.columns.astype(str)

    rows = []
    for model in models:
        print(f"  Model: {model}")
        shap_values, hps, auc_per_fold, _ = machine_learning_exp(model, X, Y)

        shap_dir = os.path.join(out_root, "shap", f"subject_{subject}")
        os.makedirs(shap_dir, exist_ok=True)
        np.savez(
            os.path.join(shap_dir, f"shap_values_{model}_subject_{subject}.npz"),
            folder_1=shap_values[0],
            folder_2=shap_values[1],
            folder_3=shap_values[2],
            folder_4=shap_values[3],
            folder_5=shap_values[4],
        )

        for fold, (auc, hp) in enumerate(zip(auc_per_fold, hps), start=1):
            rows.append(
                {"subject": subject, "model": model, "fold": fold, "AUC": auc, "C": hp}
            )

        mean_auc = float(np.mean(auc_per_fold))
        std_auc = float(np.std(auc_per_fold))
        print(f"    {model.upper()} mean AUC = {mean_auc:.3f} +/- {std_auc:.3f}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=SUBJECTS,
        help="subject indices to run (default: 0..7)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODELS),
        choices=list(MODELS),
        help="models to train (default: svc lda)",
    )
    parser.add_argument(
        "--out-root",
        default="outputs",
        help="output root directory (default: outputs/)",
    )
    args = parser.parse_args()

    auc_dir = os.path.join(args.out_root, "auc")
    os.makedirs(auc_dir, exist_ok=True)
    auc_csv = os.path.join(auc_dir, "auc_per_subject.csv")

    all_rows = []
    if os.path.exists(auc_csv):
        existing = pd.read_csv(auc_csv)
        all_rows = existing.to_dict("records")

    for subject in args.subjects:
        rows = run_subject(subject, args.models, args.out_root)
        # Replace any pre-existing rows for this (subject, model)
        keep = [
            r
            for r in all_rows
            if not (r["subject"] == subject and r["model"] in args.models)
        ]
        all_rows = keep + rows
        pd.DataFrame(all_rows).to_csv(auc_csv, index=False)
        print(f"  -> wrote {auc_csv}")


if __name__ == "__main__":
    main()
