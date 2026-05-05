"""
Figure 2: Spatial distribution of SHAP-derived relevance per subject.

Renders the brain montage (HbO top row, HbR bottom row) for all 8 subjects
using SHAP values produced by run_decoding.py.

Usage:
    python fig_2.py                # SVC (default)
    python fig_2.py --model lda
"""
import argparse
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from mne_nirs.channels import picks_pair_to_idx
from torch.nn import Conv2d

from audio_data import final_audio_data


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["svc", "lda"], default="svc")
parser.add_argument("--shap-root", default="outputs/shap")
parser.add_argument("--figures-dir", default="figures")
_args = parser.parse_args()
MODEL = _args.model
out_folder = _args.shap_root
figures_folder = _args.figures_dir
os.makedirs(figures_folder, exist_ok=True)


def fixed_attention_maps(map_type='threshold'):

    size = 7
    # Source-detector pairs for each region of interest (paper Section II.B).
    IFG_ = [[1, 1], [2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [5, 1], [5, 2]]
    Auditory_A_left = [[6, 3], [6, 4], [8, 3], [8, 4]]
    Auditory_A_right = [[10, 8], [10, 9], [11, 8], [11, 9]]
    Auditory_B_left = [[7, 5], [7, 6], [8, 5], [8, 6], [8, 7], [9, 5], [9, 6], [9, 7]]
    Auditory_B_right = [[11, 10], [11, 11], [11, 12], [12, 10], [12, 11], [13, 10], [13, 11], [13, 12]]
    Visual = [[14, 13], [14, 14], [14, 15], [15, 13], [15, 14], [16, 14], [16, 15]]
    Auditory_left = Auditory_A_left + Auditory_B_left
    Auditory_right = Auditory_A_right + Auditory_B_right

    t_max = 18.0
    fig, axes = plt.subplots(2, 8, figsize=(22, 6))

    for subject in range(8):
        idx = subject
        cmap_ticks = [i / 10 for i in range(0, 11)]

        # Load SHAP values produced by run_decoding.py (5 outer-fold arrays).
        audio_folder = os.path.join(out_folder, f'subject_{subject}')
        weights_path = f'shap_values_{MODEL}_subject_{subject}.npz'
        weight_path = os.path.join(audio_folder, weights_path)
        weights = np.load(weight_path)

        # Re-derive epochs to recover surviving channel info after SCI rejection.
        X, _, epochs = final_audio_data(subject, 0.0, t_max, None)
        epochs.pick(picks='hbo')

        # Average SHAP across all 5 outer CV folds.
        all_weights = []
        for fold_key in weights.files:
            fold_weights = weights[fold_key]
            fold_weights = np.mean(fold_weights, axis=0)
            fold_weights = fold_weights.reshape(X.shape[1], X.shape[2])
            all_weights.append(fold_weights)
        all_weights = np.stack(all_weights, axis=0)
        avg_shap = np.mean(all_weights, axis=0)

        # Min-max normalize so all subjects share the same [0, 1] colorbar.
        avg_shap = (avg_shap - np.min(avg_shap)) / (np.max(avg_shap) - np.min(avg_shap))

        # Split the channel axis: first half is HbO, second half is HbR.
        n_channels_half = X.shape[1] // 2
        hbo_shap = avg_shap[:n_channels_half, :]
        hbr_shap = avg_shap[n_channels_half:, :]

        # Reshape for Conv2d: (1, 1, n_channels, n_times).
        tensor_hbo = torch.tensor(hbo_shap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_hbr = torch.tensor(hbr_shap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 1-second box-car (data is sampled at 3.9 Hz; (1, 4) conv approximates 1 s).
        cnn = Conv2d(1, 1, kernel_size=(1, 4), stride=(1, 4), padding=(0, 1), padding_mode='replicate')
        cnn.weight.data.fill_(1.0 / 4.0)
        cnn.bias.data.fill_(0.0)

        smoothed_hbo = cnn(tensor_hbo).squeeze(0).squeeze(0).detach().numpy()
        smoothed_hbr = cnn(tensor_hbr).squeeze(0).squeeze(0).detach().numpy()

        # Sum across time -> one cumulative-SHAP value per channel.
        hbo_weights = np.sum(smoothed_hbo, axis=1)
        hbr_weights = np.sum(smoothed_hbr, axis=1)
        
        
        groups = dict(
            IFG=picks_pair_to_idx(epochs, IFG_, on_missing='ignore'),
            Auditory_A_left=picks_pair_to_idx(epochs, Auditory_A_left,on_missing='ignore'),
            Auditory_A_right=picks_pair_to_idx(epochs, Auditory_A_right,on_missing='ignore'),
            Auditory_B_left=picks_pair_to_idx(epochs, Auditory_B_left,on_missing='ignore'),
            Auditory_B_right=picks_pair_to_idx(epochs, Auditory_B_right,on_missing='ignore'),
            Visual=picks_pair_to_idx(epochs, Visual,on_missing='ignore'),
            Auditory_left=picks_pair_to_idx(epochs, Auditory_left,on_missing='ignore'),
            Auditory_right=picks_pair_to_idx(epochs, Auditory_right,on_missing='ignore'))
    
        
        new_epochs = epochs.copy()
        IFG = groups["IFG"]
        new_epochs.pick(picks=IFG)
        

        auditive_epochs = epochs.copy()
        index_auditory_left = groups["Auditory_left"]
        auditive_epochs.pick(picks=index_auditory_left)


        auditive_epochs_right = epochs.copy()
        index_auditory_right = groups["Auditory_right"]
        auditive_epochs_right.pick(picks=index_auditory_right)

        visual_epochs = epochs.copy()
        if groups["Visual"] and len(groups["Visual"]) >= 2:
            visual_epochs.pick(picks=groups["Visual"])

        # Channels that survived SCI > 0.8 in audio_data.py but still have noisy
        # topomap geometry (e.g. isolated detector after siblings dropped).
        # These per-subject manual drops keep the topomap interpolation well-conditioned.
        if subject == 0:
            new_epochs.drop_channels('S4_D1 hbo')
            auditive_epochs_right.drop_channels('S11_D10 hbo')
        elif subject == 1:
            new_epochs.drop_channels('S4_D1 hbo')
        elif subject == 2:
            new_epochs.drop_channels('S4_D1 hbo')
            auditive_epochs.drop_channels('S7_D6 hbo')
            auditive_epochs_right.drop_channels('S11_D10 hbo')
        elif subject == 3:
            new_epochs.drop_channels('S4_D1 hbo')
            auditive_epochs.drop_channels('S7_D6 hbo')
        elif subject == 4:
            auditive_epochs.drop_channels('S7_D6 hbo')
        elif subject == 5:
            auditive_epochs.drop_channels('S8_D5 hbo')
            auditive_epochs_right.drop_channels('S11_D10 hbo')
        elif subject == 6:
            new_epochs.drop_channels('S4_D1 hbo')
            auditive_epochs.drop_channels('S7_D6 hbo')
            auditive_epochs_right.drop_channels('S11_D10 hbo')
        elif subject == 7:
            auditive_epochs_right.drop_channels('S12_D11 hbo')

        hbo_1s = hbo_weights
        hbr_1s = hbr_weights
        
        
        if groups["Visual"] and len(groups["Visual"]) >= 2:

            print("len of visual group:",len(groups["Visual"]))
            min_values = [np.min(hbo_1s[IFG]), np.min(hbo_1s[index_auditory_left]), np.min(hbo_1s[index_auditory_right]), np.min(hbo_1s[groups["Visual"]])]
            max_values = [np.max(hbo_1s[IFG]), np.max(hbo_1s[index_auditory_left]), np.max(hbo_1s[index_auditory_right]), np.max(hbo_1s[groups["Visual"]])]
            min_ind = np.argmin(min_values)
            max_ind = np.argmax(max_values)

            min_val = min_values[min_ind]
            max_val = max_values[max_ind]
            
        else:

            min_values = [np.min(hbo_1s[IFG]), np.min(hbo_1s[index_auditory_left]), np.min(hbo_1s[index_auditory_right])]
            max_values = [np.max(hbo_1s[IFG]), np.max(hbo_1s[index_auditory_left]), np.max(hbo_1s[index_auditory_right])]
            min_ind = np.argmin(min_values)
            max_ind = np.argmax(max_values)

            min_val = min_values[min_ind]
            max_val = max_values[max_ind]

        #Normalize each subgroup

        hbo_1s[IFG] = (hbo_1s[IFG] - min_val) / (max_val - min_val)
        hbo_1s[index_auditory_left] = (hbo_1s[index_auditory_left] - min_val) / (max_val - min_val)
        hbo_1s[index_auditory_right] = (hbo_1s[index_auditory_right] - min_val) / (max_val - min_val)
        
        if groups["Visual"] and len(groups["Visual"]) >= 2:
            hbo_1s[groups["Visual"]] = (hbo_1s[groups["Visual"]] - min_val) / (max_val - min_val)

       
        #weights of each region
        hbo_ifg = hbo_1s[IFG]
        hbo_auditory_left = hbo_1s[index_auditory_left]
        hbo_auditory_right = hbo_1s[index_auditory_right]
        if groups["Visual"] and len(groups["Visual"]) >= 2:
            hbo_visual = hbo_1s[groups["Visual"]]

        # Drop the same SHAP-weight slot for each channel removed above, so the
        # weight vector aligns with the post-drop epochs.info used by plot_topomap.
        if subject == 0:
            hbo_ifg = np.delete(hbo_ifg, 4)
            hbo_auditory_right = np.delete(hbo_auditory_right, 4)
        elif subject == 1:
            hbo_ifg = np.delete(hbo_ifg, 4)
        elif subject == 2:
            hbo_ifg = np.delete(hbo_ifg, 4)
            hbo_auditory_left = np.delete(hbo_auditory_left, 5)
            hbo_auditory_right = np.delete(hbo_auditory_right, 4)
        elif subject == 3:
            hbo_ifg = np.delete(hbo_ifg, 4)
            hbo_auditory_left = np.delete(hbo_auditory_left, 5)
        elif subject == 4:
            hbo_auditory_left = np.delete(hbo_auditory_left, 5)
        elif subject == 5:
            hbo_auditory_left = np.delete(hbo_auditory_left, 5)
            hbo_auditory_right = np.delete(hbo_auditory_right, 4)
        elif subject == 6:
            hbo_ifg = np.delete(hbo_ifg, 4)
            hbo_auditory_left = np.delete(hbo_auditory_left, 5)
            hbo_auditory_right = np.delete(hbo_auditory_right, 4)
        elif subject == 7:
            hbo_auditory_right = np.delete(hbo_auditory_right, 3)
            
            
        ifg_mask = np.zeros(len(hbo_ifg), dtype=bool)
        auditory_mask_left = np.zeros(len(hbo_auditory_left), dtype=bool)
        auditory_mask_right = np.zeros(len(hbo_auditory_right), dtype=bool)

        if groups["Visual"] and len(groups["Visual"]) >= 2:
            visual_mask = np.zeros(len(groups["Visual"]), dtype=bool)

        ifg_mask = hbo_ifg >= 0.8
        auditory_mask_left = hbo_auditory_left >= 0.8
        auditory_mask_right = hbo_auditory_right >= 0.8

        if groups["Visual"] and len(groups["Visual"]) >= 2:
            visual_mask = hbo_visual >= 0.8

        if subject !=7:
            img_ifg, _ = mne.viz.plot_topomap(hbo_ifg, pos=new_epochs.info, ch_type='hbo', sensors=True,
                                            res=256, axes=axes[0, idx], show=False, cmap='viridis', contours=0,
                                            extrapolate='local',size=size)

        # Plot Auditory_left weights
        img_auditory, _ = mne.viz.plot_topomap(hbo_auditory_left, pos=auditive_epochs.info, ch_type='hbo',
                                                        sensors=True, res=256, size=size, axes=axes[0, idx], show=False,
                                                        cmap='viridis', contours=0, extrapolate='local')

        # Plot Auditory_right weights
        if subject !=3:
            img_auditory_right, _ = mne.viz.plot_topomap(hbo_auditory_right, pos=auditive_epochs_right.info,
                                                                        ch_type='hbo', sensors=True, res=256, size=size,
                                                                        axes=axes[0, idx], show=False, cmap='viridis', contours=0,
                                                                        extrapolate='local')
         
         
         

        # Plot Visual weights
        if groups["Visual"] and len(groups["Visual"]) >= 2:
            img_visual, _ = mne.viz.plot_topomap(hbo_visual, pos=visual_epochs.info, ch_type='hbo',
                                                        sensors=True, res=256, size=size, axes=axes[0, idx], show=False,
                                                        cmap='viridis', contours=0, extrapolate='local')
     
        axes[0, subject].set_title(f'Sub.{subject}')

        # Subject 2 has all four regions present; we cache its handles to anchor
        # the shared HbO colorbar at the bottom of the loop.
        if subject == 2:
            hbo_imgs = [img_ifg, img_auditory, img_auditory_right, img_visual]
            max_hbo_ind = max_ind


        # ============================  HbR  ============================
        if groups["Visual"] and len(groups["Visual"]) >= 2:
            min_values = [np.min(hbr_1s[IFG]), np.min(hbr_1s[index_auditory_left]), np.min(hbr_1s[index_auditory_right]), np.min(hbr_1s[groups["Visual"]])]
            max_values = [np.max(hbr_1s[IFG]), np.max(hbr_1s[index_auditory_left]), np.max(hbr_1s[index_auditory_right]), np.max(hbr_1s[groups["Visual"]])]
            min_ind = np.argmin(min_values)
            max_ind = np.argmax(max_values)

            min_val = min_values[min_ind]
            max_val = max_values[max_ind]

        else:
            
            min_values = [np.min(hbr_1s[IFG]), np.min(hbr_1s[index_auditory_left]), np.min(hbr_1s[index_auditory_right])]
            max_values = [np.max(hbr_1s[IFG]), np.max(hbr_1s[index_auditory_left]), np.max(hbr_1s[index_auditory_right])]
            min_ind = np.argmin(min_values)
            max_ind = np.argmax(max_values)

            min_val = min_values[min_ind]
            max_val = max_values[max_ind]
            
        #Normalize each subgroup

        hbr_1s[IFG] = (hbr_1s[IFG] - min_val) / (max_val - min_val)
        hbr_1s[index_auditory_left] = (hbr_1s[index_auditory_left] - min_val) / (max_val - min_val)
        hbr_1s[index_auditory_right] = (hbr_1s[index_auditory_right] - min_val) / (max_val - min_val)
        hbr_1s[groups["Visual"]] = (hbr_1s[groups["Visual"]] - min_val) / (max_val - min_val)

        hbr_ifg = hbr_1s[IFG]
        hbr_auditory_left = hbr_1s[index_auditory_left]
        hbr_auditory_right = hbr_1s[index_auditory_right]

        if groups["Visual"] and len(groups["Visual"]) >= 2:
            hbr_visual = hbr_1s[groups["Visual"]]

        # Same drop pattern as the HbO side, mirrored on the HbR weight vector.
        if subject == 0:
            hbr_ifg = np.delete(hbr_ifg, 4)
            hbr_auditory_right = np.delete(hbr_auditory_right, 4)
        elif subject == 1:
            hbr_ifg = np.delete(hbr_ifg, 4)
        elif subject == 2:
            hbr_ifg = np.delete(hbr_ifg, 4)
            hbr_auditory_left = np.delete(hbr_auditory_left, 5)
            hbr_auditory_right = np.delete(hbr_auditory_right, 4)
        elif subject == 3:
            hbr_ifg = np.delete(hbr_ifg, 4)
            hbr_auditory_left = np.delete(hbr_auditory_left, 5)
        elif subject == 4:
            hbr_auditory_left = np.delete(hbr_auditory_left, 5)
        elif subject == 5:
            hbr_auditory_left = np.delete(hbr_auditory_left, 5)
            hbr_auditory_right = np.delete(hbr_auditory_right, 4)
        elif subject == 6:
            hbr_ifg = np.delete(hbr_ifg, 4)
            hbr_auditory_left = np.delete(hbr_auditory_left, 5)
            hbr_auditory_right = np.delete(hbr_auditory_right, 4)
        elif subject == 7:
            hbr_auditory_right = np.delete(hbr_auditory_right, 3)
            
        #Mask for each region
        hbr_mask_ifg = np.zeros(len(hbr_ifg), dtype=bool)
        hbr_mask_auditory_left = np.zeros(len(hbr_auditory_left), dtype=bool)
        hbr_mask_auditory_right = np.zeros(len(hbr_auditory_right), dtype=bool)
        if groups["Visual"] and len(groups["Visual"]) >= 2:
            hbr_mask_visual = np.zeros(len(hbr_visual), dtype=bool)

        hbr_mask_ifg = hbr_ifg >=0.8
        hbr_mask_auditory_left = hbr_auditory_left >=0.8
        hbr_mask_auditory_right = hbr_auditory_right >=0.8

        if groups["Visual"] and len(groups["Visual"]) >= 2:

            hbr_mask_visual = hbr_visual >=0.8

        

        if subject !=7:
            hbr_img_ifg, _ = mne.viz.plot_topomap(hbr_ifg, pos=new_epochs.info, ch_type='hbo', sensors=True,
                                            res=256, size=size, axes=axes[1, idx], show=False, cmap='viridis', contours=0,
                                                extrapolate='local')

        
        # Plot Auditory_left weights
        hbr_img_auditory, _ = mne.viz.plot_topomap(hbr_auditory_left, pos=auditive_epochs.info, ch_type='hbo',
                                                        sensors=True, res=256, size=size, axes=axes[1, idx], show=False,
                                                        cmap='viridis', contours=0, extrapolate='local')

        # Plot Auditory_right weights
        if subject !=3:
            hbr_img_auditory_right, _ = mne.viz.plot_topomap(hbr_auditory_right, pos=auditive_epochs_right.info,
                                                                        ch_type='hbo', sensors=True, res=256, size=size,
                                                                        axes=axes[1, idx], show=False, cmap='viridis', contours=0,
                                                                        extrapolate='local')

        # Plot Visual weights
        if groups["Visual"] and len(groups["Visual"]) >= 2:

            hbr_img_visual, _ = mne.viz.plot_topomap(hbr_visual, pos=visual_epochs.info, ch_type='hbo',
                                                    sensors=True, res=256, size=size, axes=axes[1, idx], show=False,
                                                    cmap='viridis', contours=0, extrapolate='local')
                
       
       
        axes[1, idx].set_title(f'Sub.{subject}')

        if subject == 2:
            hbr_imgs = [hbr_img_ifg, hbr_img_auditory, hbr_img_auditory_right, hbr_img_visual]
            max_hbr_ind = max_ind

        print("Subject:", subject)

    # Shared HbO colorbar at the top, HbR colorbar at the bottom.
    cbar_hbo = fig.colorbar(hbo_imgs[max_hbo_ind], ax=axes[0, :], shrink=0.75,
                            orientation='vertical', ticks=cmap_ticks)
    cbar_hbo.set_ticklabels(cmap_ticks)
    cbar_hbo.set_label('HbO')

    cbar_hbr = plt.colorbar(hbr_imgs[max_hbr_ind], ax=axes[1, :], shrink=0.75,
                            orientation='vertical', ticks=cmap_ticks)
    cbar_hbr.set_ticklabels(cmap_ticks)
    cbar_hbr.set_label('HbR')

    # Vertical "SHAP" label between the two rows, alongside the colorbars.
    fig.text(0.81, 0.5, 'SHAP', va='center', ha='center', rotation='vertical', fontsize=16)

    plt.savefig(os.path.join(figures_folder, f'fig2_brain_montage_{MODEL}.png'), dpi=500)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    fixed_attention_maps(map_type='threshold')