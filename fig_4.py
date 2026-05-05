"""
Figure 4: Modified Multiscale Entropy (MME) of fNIRS signals.

Reproduces the two panels of paper Fig. 4:
  figures/fig4a_mme_per_subject.png   per-subject MME curves across scales
                                      (paper Fig. 4a)
  figures/fig4b_pc1_scores.png        first principal component of per-subject
                                      MME curves, by performance group
                                      (paper Fig. 4b)

Method (paper Section II.G):
  Sample entropy with embedding m=2, time delay tau, tolerance r=0.2*sd.
  Coarse-graining via moving average for tau in [1, MAX_SCALE].

Computed on silence trials, HbO only, for the 6 brain areas defined in the
paper. Per-subject MME = mean across (trial, channel, area).

Run:
    python fig_4.py                # uses defaults, takes ~10 min
    python fig_4.py --max-scale 30
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_nirs.channels import picks_pair_to_idx
from scipy.stats import ttest_ind

from audio_data import final_audio_data


# Source-detector pairs (same as paper Section II.B)
IFG = [[1, 1], [2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [5, 1], [5, 2]]
AUDITORY_A_LEFT = [[6, 3], [6, 4], [8, 3], [8, 4]]
AUDITORY_A_RIGHT = [[10, 8], [10, 9], [11, 8], [11, 9]]
AUDITORY_B_LEFT = [[7, 5], [7, 6], [8, 5], [8, 6], [8, 7], [9, 5], [9, 6], [9, 7]]
AUDITORY_B_RIGHT = [[11, 10], [11, 11], [11, 12], [12, 10], [12, 11], [13, 10], [13, 11], [13, 12]]
VISUAL = [[14, 13], [14, 14], [14, 15], [15, 13], [15, 14], [16, 14], [16, 15]]


def sample_entropy_delay(x, m=2, r=0.2, delay=1):
    """Sample entropy on series `x` with embedding dim `m` and time delay `delay`."""
    x = np.asarray(x, float)
    n = len(x)
    tol = r * np.std(x, ddof=1)
    if tol == 0:
        return 0.0

    def templates(dim):
        size = n - (dim - 1) * delay
        if size <= 1:
            return None
        idx = np.arange(size)[:, None] + delay * np.arange(dim)[None, :]
        return x[idx]

    xm = templates(m)
    xm1 = templates(m + 1)
    if xm is None or xm1 is None:
        return np.nan

    def count(arr):
        total = 0
        for i in range(len(arr) - 1):
            d = np.max(np.abs(arr[i + 1:] - arr[i]), axis=1)
            total += np.sum(d <= tol)
        return total

    b = count(xm)
    a = count(xm1)
    if b == 0:
        return np.nan
    if a == 0:
        return np.inf
    return -np.log(a / b)


def coarse_grain_overlap(x, scale):
    """Moving-average coarse-graining of length `scale`."""
    if scale < 1 or scale > len(x):
        raise ValueError("scale must be in [1, len(x)]")
    n_out = len(x) - scale + 1
    y = np.empty(n_out)
    for i in range(n_out):
        y[i] = np.mean(x[i:i + scale])
    return y


def mmse(x, max_scale=20, m=2, r=0.2):
    """Modified multiscale (sample) entropy across scales 1..max_scale."""
    x = np.asarray(x, float)
    out = []
    for tau in range(1, max_scale + 1):
        y = coarse_grain_overlap(x, tau)
        out.append(sample_entropy_delay(y, m=m, r=r, delay=tau))
    return np.array(out)


def compute_entropy_dataframe(subjects, t_min, t_max, max_scale, m, r):
    """Return a long-format DataFrame with one row per (subject, trial, area, channel)."""
    rows = []
    silence_label = 1  # encoding from final_audio_data: -1=Audio, +1=Control(silence)
    for subject in subjects:
        x_arr, y, epochs = final_audio_data(subject, t_min, t_max, chroma='hbo')
        sil_idx = np.where(np.array(y) == silence_label)[0]
        if sil_idx.size == 0:
            continue
        x_sil = x_arr[sil_idx]

        groups = dict(
            IFG=picks_pair_to_idx(epochs, IFG, on_missing='ignore'),
            Auditory_A_left=picks_pair_to_idx(epochs, AUDITORY_A_LEFT, on_missing='ignore'),
            Auditory_A_right=picks_pair_to_idx(epochs, AUDITORY_A_RIGHT, on_missing='ignore'),
            Auditory_B_left=picks_pair_to_idx(epochs, AUDITORY_B_LEFT, on_missing='ignore'),
            Auditory_B_right=picks_pair_to_idx(epochs, AUDITORY_B_RIGHT, on_missing='ignore'),
            Visual=picks_pair_to_idx(epochs, VISUAL, on_missing='ignore'),
        )

        for area, ch_indices in groups.items():
            for ch in ch_indices:
                for trial in range(x_sil.shape[0]):
                    sig = x_sil[trial, ch, :]
                    rows.append({
                        'subject': subject,
                        'trial': trial,
                        'area': area,
                        'channel': int(ch),
                        'MME': mmse(sig, max_scale=max_scale, m=m, r=r),
                    })
        print(f"  subject {subject}: {len(rows)} rows so far")

    return pd.DataFrame(rows)


def plot_fig4a(entropy_df, save_path, low_group):
    """Figure 4a: per-subject MME curves averaged across all (trial, channel, area)."""
    plt.figure(figsize=(10, 6))
    group_colors = {'low': '#d62728', 'high': '#1f77b4'}
    for subject in sorted(entropy_df['subject'].unique()):
        subj = entropy_df[entropy_df['subject'] == subject]
        mme_arrays = np.vstack(subj['MME'].values)
        mme_mean = np.nanmean(mme_arrays, axis=0)
        color = group_colors['low'] if subject in low_group else group_colors['high']
        x_vals = range(1, len(mme_mean) + 1)
        plt.plot(x_vals, mme_mean, marker='o', label=f'Subject {int(subject)}',
                 color=color, linewidth=2, alpha=0.85)
        plt.text(x_vals[0] - 0.3, mme_mean[0], f'S{int(subject)}', fontsize=8,
                 color=color, va='center', ha='right', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=color, alpha=0.8))
    plt.xlabel('Scale', fontsize=18)
    plt.ylabel('Mean MME', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()


def plot_fig4b(entropy_df, save_path, low_idx, high_idx):
    """Figure 4b: first principal component of per-subject MME curves, by group."""
    avg_mme_df = (
        entropy_df
        .groupby('subject')['MME']
        .apply(lambda x: np.nanmean(np.vstack(x), axis=0))
        .reset_index()
    )
    avg_mme_df.columns = ['subject', 'avg_MME']
    subj_curve = np.vstack(avg_mme_df['avg_MME'].to_numpy())
    subj_curve[np.isinf(subj_curve)] = np.nan

    # Drop scales (columns) that are entirely NaN — typically the highest scales
    # where the coarse-grained series is too short for some subjects.
    valid_cols = ~np.all(np.isnan(subj_curve), axis=0)
    subj_curve = subj_curve[:, valid_cols]
    n_dropped = (~valid_cols).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} all-NaN scale(s) before SVD")

    # Center across subjects, then fill any remaining per-subject NaNs with the
    # column (across-subject) mean so SVD is well-defined.
    centered = subj_curve - np.nanmean(subj_curve, axis=0, keepdims=True)
    col_means = np.nanmean(centered, axis=0)
    nan_mask = np.isnan(centered)
    if nan_mask.any():
        centered[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = u[:, 0]
    print(f"  Variance explained by PC1: {s[0] ** 2 / np.sum(s ** 2):.3f}")

    x_low = pc1[low_idx]
    x_high = pc1[high_idx]
    t_stat, p_val = ttest_ind(x_low, x_high)
    print(f"  PC1 two-sided t-test (low vs high): t={t_stat:.3f}, p={p_val:.4f}")

    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(low_idx):
        plt.scatter(x_low[i], 0, s=120, color='red', alpha=0.85)
        plt.text(x_low[i], -0.025, f'S{int(idx)}', fontsize=10, ha='center',
                 va='top', fontweight='bold', color='red',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='red', alpha=0.9))
    for i, idx in enumerate(high_idx):
        plt.scatter(x_high[i], 0, s=120, color='blue', alpha=0.85)
        plt.text(x_high[i], 0.025, f'S{int(idx)}', fontsize=10, ha='center',
                 va='bottom', fontweight='bold', color='blue',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='blue', alpha=0.9))
    plt.ylim(-0.08, 0.08)
    plt.yticks([])
    plt.xlabel('PC1 Score', fontsize=14)
    plt.title(f'PC1 of subject MME curves '
              f'(t={t_stat:.2f}, p={p_val:.3f})', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--t-min', type=float, default=0.0)
    parser.add_argument('--t-max', type=float, default=18.0)
    parser.add_argument('--max-scale', type=int, default=20,
                        help='Number of MME scales (paper uses 20).')
    parser.add_argument('--m', type=int, default=2,
                        help='Embedding dimension (paper uses 2).')
    parser.add_argument('--r', type=float, default=0.2,
                        help='Tolerance multiplier (paper uses 0.2).')
    parser.add_argument('--low-subjects', type=int, nargs='+', default=[2, 3, 7],
                        help='Indices of low-performing subjects (paper: 2, 3, 7).')
    parser.add_argument('--figures-dir', default='figures')
    parser.add_argument('--cache-csv', default='outputs/mme/entropy_df.pkl',
                        help='Cache MME computation here. Delete to recompute.')
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.cache_csv), exist_ok=True)

    if os.path.exists(args.cache_csv):
        print(f"Loading cached entropy_df from {args.cache_csv}")
        entropy_df = pd.read_pickle(args.cache_csv)
    else:
        print("Computing MME for all (subject, trial, area, channel)...")
        entropy_df = compute_entropy_dataframe(
            args.subjects, args.t_min, args.t_max,
            args.max_scale, args.m, args.r,
        )
        entropy_df.to_pickle(args.cache_csv)
        print(f"  cached to {args.cache_csv}")

    low_set = set(args.low_subjects)
    all_subjects = set(int(s) for s in entropy_df['subject'].unique())
    high_idx = np.array(sorted(all_subjects - low_set))
    low_idx = np.array(sorted(low_set))

    plot_fig4a(entropy_df, os.path.join(args.figures_dir, 'fig4a_mme_per_subject.png'),
               low_group=low_set)
    print(f"  -> {os.path.join(args.figures_dir, 'fig4a_mme_per_subject.png')}")

    plot_fig4b(entropy_df, os.path.join(args.figures_dir, 'fig4b_pc1_scores.png'),
               low_idx=low_idx, high_idx=high_idx)
    print(f"  -> {os.path.join(args.figures_dir, 'fig4b_pc1_scores.png')}")


if __name__ == '__main__':
    main()
