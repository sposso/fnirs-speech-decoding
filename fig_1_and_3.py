"""
Figure 1 (subject-wise SHAP) and Figure 3 (group-averaged SHAP) from the paper.

Reads SHAP values produced by run_decoding.py and writes:
  figures/<MODEL>_brain_area_per_subject.png        (paper Fig 1a/1b)
  figures/<MODEL>_time_points_per_subject.png       (paper Fig 1c/1d)
  figures/<MODEL>_brain_area_group.png              (paper Fig 3a/3b)
  figures/<MODEL>_time_points_group.png             (paper Fig 3c/3d)

Usage:
    python fig_1_and_3.py                # SVC (default)
    python fig_1_and_3.py --model lda    # LDA
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import Conv2d

from audio_data import final_audio_data


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["svc", "lda"], default="svc")
parser.add_argument("--shap-root", default="outputs/shap")
parser.add_argument("--figures-dir", default="figures")
args = parser.parse_args()
MODEL = args.model
out_folder = args.shap_root
folder_ = args.figures_dir
os.makedirs(folder_, exist_ok=True)

# Strip "S", "D", "hbo" and spaces from MNE channel names so "S1_D1 hbo" -> "1_1".
clean_ch_name = str.maketrans("", "", "DShbo ")

# Per-subject cumulative SHAP DataFrames keyed by brain area (HbO and HbR).
hbo_area_dfs_per_subject = []
hbr_area_dfs_per_subject = []

# Channels covering each brain region, encoded as "source_detector" strings.
# (E.g. "1_1" means source 1 is paired with detector 1.) The mapping comes
# from the supplementary materials of the dataset paper (Shader et al.).
channels_to_brain_areas = {
    "IFG":    ["1_1", "2_1", "3_1", "3_2", "4_1", "4_2", "5_1", "5_2"],
    "Au_A_L": ["6_3", "6_4", "8_3", "8_4"],
    "Au_A_R": ["10_8", "10_9", "11_8", "11_9"],
    "Au_B_L": ["7_5", "7_6", "8_5", "8_6", "8_7", "9_5", "9_6", "9_7"],
    "Au_B_R": ["11_10", "11_11", "11_12", "12_10", "12_11", "13_10", "13_11", "13_12"],
    "V_A":    ["14_13", "14_14", "14_15"],
    "V_B":    ["15_13", "15_14", "16_14", "16_15"],
}

# Cumulative SHAP per (subject, second). Used for Fig 1c/1d and Fig 3c/3d.
n_subjects = 8
n_seconds = 18  # 0..18s window after the conv smoothing
cum_shap_time_hbo = np.zeros((n_subjects, n_seconds))
cum_shap_time_hbr = np.zeros((n_subjects, n_seconds))

subjects = list(range(n_subjects))

for subject_idx, subject in enumerate(subjects):
    # Load SHAP values produced by run_decoding.py for this (subject, model).
    audio_folder = os.path.join(out_folder, f'subject_{subject}')
    weights_path = f'shap_values_{MODEL}_subject_{subject}.npz'
    weight_path = os.path.join(audio_folder, weights_path)
    weights = np.load(weight_path)

    # Re-load the epochs to get the actual channel names that survived SCI rejection.
    t_max = 18
    X, _, epochs = final_audio_data(subject, 0.0, t_max, None)
    ch_names = epochs.ch_names
    hbo_names_o = [name.translate(clean_ch_name) for name in ch_names if 'hbo' in name]

    # Per-fold buffers (5 outer CV folds).
    n_folds = 5
    n_channels_half = X.shape[1] // 2  # HbO half (HbR is the other half)
    fold_shap_time_hbo = np.zeros((n_folds, n_seconds))
    fold_shap_time_hbr = np.zeros((n_folds, n_seconds))
    fold_shap_channel_hbo = np.zeros((n_folds, n_channels_half))
    fold_shap_channel_hbr = np.zeros((n_folds, n_channels_half))

    for fold_idx, fold_key in enumerate(weights.files):
        # SHAP values for this fold: (n_test, n_channels * n_times) -> mean across instances.
        fold_weights = weights[fold_key]
        fold_weights = np.mean(fold_weights, axis=0)
        fold_weights = fold_weights.reshape(X.shape[1], X.shape[2])

        # Min-max normalize within this fold so subjects/folds are comparable.
        fold_weights = (fold_weights - np.min(fold_weights)) / (np.max(fold_weights) - np.min(fold_weights))

        # Split the channel axis: first half is HbO, second half is HbR.
        hbo_weights = fold_weights[:n_channels_half, :]
        hbr_weights = fold_weights[n_channels_half:, :]

        # Reshape for Conv2d: (1, 1, n_channels, n_times)
        tensor_hbo = torch.tensor(hbo_weights, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_hbr = torch.tensor(hbr_weights, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # The fNIRS data is sampled at 3.9 Hz, so a (1, 4) box-car average
        # downsamples each ~4 samples to a ~1-second resolution.
        cnn = Conv2d(1, 1, kernel_size=(1, 4), stride=(1, 4), padding=(0, 1), padding_mode='replicate')
        cnn.weight.data.fill_(1.0 / 4.0)
        cnn.bias.data.fill_(0.0)

        smoothed_hbo = cnn(tensor_hbo).squeeze(0).squeeze(0).detach().numpy()
        smoothed_hbr = cnn(tensor_hbr).squeeze(0).squeeze(0).detach().numpy()

        # Cumulative SHAP per second (sum across channels).
        fold_shap_time_hbo[fold_idx] = np.sum(smoothed_hbo, axis=0)
        fold_shap_time_hbr[fold_idx] = np.sum(smoothed_hbr, axis=0)

        # Cumulative SHAP per channel (sum across time).
        fold_shap_channel_hbo[fold_idx] = np.sum(smoothed_hbo, axis=1)
        fold_shap_channel_hbr[fold_idx] = np.sum(smoothed_hbr, axis=1)

    # Aggregate folds: sum into a single (n_seconds,) and (n_channels_half,) vector per subject.
    cum_shap_time_hbo[subject_idx] = np.sum(fold_shap_time_hbo, axis=0)
    cum_shap_time_hbr[subject_idx] = np.sum(fold_shap_time_hbr, axis=0)
    subj_channel_hbo = np.sum(fold_shap_channel_hbo, axis=0)
    subj_channel_hbr = np.sum(fold_shap_channel_hbr, axis=0)

    # Map each surviving channel to its brain area; assign 0 to channels rejected by SCI.
    rows_hbo, rows_hbr = [], []
    for brain_area, channel_ids in channels_to_brain_areas.items():
        for ch_id in channel_ids:
            if ch_id in hbo_names_o:
                index = hbo_names_o.index(ch_id)
                rows_hbo.append((ch_id, brain_area, subj_channel_hbo[index]))
                rows_hbr.append((ch_id, brain_area, subj_channel_hbr[index]))
            else:
                rows_hbo.append((ch_id, brain_area, 0))
                rows_hbr.append((ch_id, brain_area, 0))

    # Sum over channels within each brain area -> one value per area per subject.
    df_hbo = pd.DataFrame(rows_hbo, columns=['Channel Name', 'Brain Area', 'Count'])
    df_hbo_combined = df_hbo.groupby('Brain Area').agg({'Count': 'sum'}).reset_index()
    df_hbr = pd.DataFrame(rows_hbr, columns=['Channel Name', 'Brain Area', 'Count'])
    df_hbr_combined = df_hbr.groupby('Brain Area').agg({'Count': 'sum'}).reset_index()
    hbo_area_dfs_per_subject.append(df_hbo_combined)
    hbr_area_dfs_per_subject.append(df_hbr_combined)


# --- Figure 1c/1d: per-subject cumulative SHAP per second ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, (subj_time_hbo, subj_time_hbr) in enumerate(zip(cum_shap_time_hbo, cum_shap_time_hbr)):
    ax = axes[i]
    bar_width = 0.35
    index = np.arange(subj_time_hbo.shape[0])
    ax.bar(index, subj_time_hbo, bar_width, label='HbO', color='b')
    ax.bar(index + bar_width, subj_time_hbr, bar_width, label='HbR', color='g')
    ax.set_title(f'sub.{i}', fontsize=23)
    sparse_ticks = [1, 5, 10, 15, 17]
    ax.set_xticks(sparse_ticks)
    ax.set_xticklabels(sparse_ticks, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig.text(0.5, 0.02, 'Time (s)', ha='center', va='center', fontsize=25)
fig.text(0.03, 0.5, 'Cumulative SHAP values', ha='center', va='center', rotation='vertical', fontsize=24)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=23, bbox_to_anchor=(0.52, 1.03))
plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
plt.savefig(os.path.join(folder_, f'{MODEL}_time_points_per_subject.png'), dpi=500)
plt.show()


# --- Figure 3c/3d: group mean +/- std cumulative SHAP per second ---
group_time_hbo_mean = np.mean(cum_shap_time_hbo, axis=0)
group_time_hbo_std = np.std(cum_shap_time_hbo, axis=0)
group_time_hbr_mean = np.mean(cum_shap_time_hbr, axis=0)
group_time_hbr_std = np.std(cum_shap_time_hbr, axis=0)

plt.figure(figsize=(12, 8))
bar_width = 0.35
index = np.arange(group_time_hbo_mean.shape[0])
plt.bar(index, group_time_hbo_mean, bar_width, yerr=group_time_hbo_std, label='HbO', color='b', capsize=5)
plt.bar(index + bar_width, group_time_hbr_mean, bar_width, yerr=group_time_hbr_std, label='HbR', color='g', capsize=5)
plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Average SHAP values', fontsize=25)
plt.xticks(ticks=np.arange(0, 18, 1), fontsize=17)
plt.yticks(fontsize=17)
plt.ylim(0, 110)
plt.legend(fontsize=23, loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(folder_, f'{MODEL}_time_points_group.png'), dpi=500)
plt.show()


# --- Figure 3a/3b: group mean +/- std cumulative SHAP per brain area ---
hbo_areas_all = pd.concat(hbo_area_dfs_per_subject, ignore_index=True)
hbr_areas_all = pd.concat(hbr_area_dfs_per_subject, ignore_index=True)

group_area_hbo_mean = hbo_areas_all.groupby('Brain Area').agg({'Count': 'mean'}).reset_index()
group_area_hbo_std = hbo_areas_all.groupby('Brain Area').agg({'Count': 'std'}).reset_index()
group_area_hbr_mean = hbr_areas_all.groupby('Brain Area').agg({'Count': 'mean'}).reset_index()
group_area_hbr_std = hbr_areas_all.groupby('Brain Area').agg({'Count': 'std'}).reset_index()



plt.figure(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(group_area_hbo_mean['Brain Area']))

plt.bar(index, group_area_hbo_mean['Count'], bar_width,
        yerr=group_area_hbo_std['Count'],
        label='HbO', color='tab:brown', capsize=5, error_kw=dict(lw=3))
plt.bar(index + bar_width, group_area_hbr_mean['Count'], bar_width,
        yerr=group_area_hbr_std['Count'],
        label='HbR', color='orange', capsize=5, error_kw=dict(lw=3))

plt.xlabel('Brain Area', fontsize=25)
plt.ylabel('Average SHAP values', fontsize=24)
plt.xticks(ticks=index + bar_width / 2,
           labels=group_area_hbo_mean['Brain Area'], rotation=60, fontsize=25)
plt.yticks(fontsize=17)
plt.ylim(0, 400)
plt.legend(fontsize=23)
plt.tight_layout()
plt.savefig(os.path.join(folder_, f'{MODEL}_brain_area_group.png'), dpi=500)
plt.show()


# --- Figure 1a/1b: per-subject cumulative SHAP per brain area ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, (df_hbo, df_hbr) in enumerate(zip(hbo_area_dfs_per_subject, hbr_area_dfs_per_subject)):
    ax = axes[i]
    bar_width = 0.35
    index = np.arange(len(df_hbo['Brain Area']))
    ax.bar(index, df_hbo['Count'], bar_width, label='HbO', color='tab:brown')
    ax.bar(index + bar_width, df_hbr['Count'], bar_width, label='HbR', color='orange')
    ax.set_title(f'sub.{i}', fontsize=23)
    ax.tick_params(axis='x', rotation=60, labelsize=18)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df_hbo['Brain Area'])
    ax.tick_params(axis='y', labelsize=18)

fig.text(0.5, 0.02, 'Brain Area', ha='center', va='center', fontsize=23)
fig.text(0.03, 0.5, 'Cumulative SHAP values', ha='center', va='center',
         rotation='vertical', fontsize=23)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=23, bbox_to_anchor=(0.52, 1.03))
plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
plt.savefig(os.path.join(folder_, f'{MODEL}_brain_area_per_subject.png'), dpi=500)
plt.show()



# %%
