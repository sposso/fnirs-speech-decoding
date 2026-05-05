"""
Per-subject statistical analysis of SHAP values.

Supports Table II in the paper. For each of the 8 subjects, ranks the 6 brain
areas by mean SHAP relevance, runs a Kruskal-Wallis omnibus test, then pairwise
Mann-Whitney U comparisons with effect sizes. Reads SHAP values written by
run_decoding.py.

Areas:
  IFG, Auditory_A_left, Auditory_A_right, Auditory_B_left, Auditory_B_right, Visual

Usage:
    python table_2.py                # SVC by default
    python table_2.py --model lda
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from scipy.stats import (kruskal, mannwhitneyu, shapiro)
from statsmodels.stats.multitest import multipletests
from torch.nn import Conv2d

from audio_data import final_audio_data

warnings.filterwarnings('ignore')


class PerSubjectShapAnalysis:
    """
    Perform per-subject and across-subject statistical analysis of SHAP values.
    """
    
    def __init__(self, shap_folder='outputs/shap', output_folder='outputs/shap_analysis', alpha=0.05):
        self.shap_folder = shap_folder
        self.output_folder = output_folder
        self.alpha = alpha
        os.makedirs(output_folder, exist_ok=True)

        # Source-detector pair definitions (used by mne_nirs.picks_pair_to_idx if needed)
        self.brain_areas = {
            'IFG': [[1, 1], [2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [5, 1], [5, 2]],
            'Auditory_A_left': [[6, 3], [6, 4], [8, 3], [8, 4]],
            'Auditory_A_right': [[10, 8], [10, 9], [11, 8], [11, 9]],
            'Auditory_B_left': [[7, 5], [7, 6], [8, 5], [8, 6], [8, 7], [9, 5], [9, 6], [9, 7]],
            'Auditory_B_right': [[11, 10], [11, 11], [11, 12], [12, 10], [12, 11], [13, 10], [13, 11], [13, 12]],
            'Visual': [[14, 13], [14, 14], [14, 15], [15, 13], [15, 14], [16, 14], [16, 15]],
        }

        # Same channels as 's_d' strings, for matching against cleaned channel names.
        self.brain_areas_mapping = {
            'IFG': ['1_1', '2_1', '3_1', '3_2', '4_1', '4_2', '5_1', '5_2'],
            'Auditory_A_left': ['6_3', '6_4', '8_3', '8_4'],
            'Auditory_A_right': ['10_8', '10_9', '11_8', '11_9'],
            'Auditory_B_left': ['7_5', '7_6', '8_5', '8_6', '8_7', '9_5', '9_6', '9_7'],
            'Auditory_B_right': ['11_10', '11_11', '11_12', '12_10', '12_11', '13_10', '13_11', '13_12'],
            'Visual': ['14_13', '14_14', '14_15', '15_13', '15_14', '16_14', '16_15'],
        }
        
    def load_shap_and_aggregate(self, subject, model='svc'):
        """
        Load SHAP values and aggregate by brain areas (using same method as graphs.py).
        
        Returns:
        --------
        area_data : dict
            Dictionary with area names as keys and (n_features,) arrays as values
        """
        audio_folder = os.path.join(self.shap_folder, f'subject_{subject}')
        weight_path = os.path.join(audio_folder, f'shap_values_{model}_subject_{subject}.npz')
        
        if not os.path.exists(weight_path):
            print(f"File not found: {weight_path}")
            return None
        
        # Load data
        weights = np.load(weight_path)
        X, _, epochs = final_audio_data(subject, 0.0, 18.0, None)
        ch_names = epochs.ch_names
        
        # Get HbO channel names (same as graphs.py)
        translation = str.maketrans("", "", "DShbo ")
        hbo_names_o = [name.translate(translation) for name in ch_names if 'hbo' in name]
        
        # Initialize aggregated data for each area
        area_data = {area: [] for area in self.brain_areas_mapping.keys()}
        
        # Process each fold
        count_channels = np.zeros(X.shape[1] // 2)
        count_channels_hbr = np.zeros(X.shape[1] // 2)
        
        for folder in sorted(weights.files):
            folder_weights = weights[folder]
            
            # Mean across instances
            folder_weights = np.mean(folder_weights, axis=0)
            
            # Reshape
            folder_weights = folder_weights.reshape(X.shape[1], X.shape[2])
            
            # Normalize
            folder_weights = (folder_weights - np.min(folder_weights)) / (np.max(folder_weights) - np.min(folder_weights))
            
            # Split HbO and HbR
            binary_hbo_weights = folder_weights[:X.shape[1]//2, :]
            binary_hbr_weights = folder_weights[X.shape[1]//2:, :]
            
            # Apply CNN convolution (same as graphs.py)
            tensor_hbo = torch.tensor(binary_hbo_weights, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensor_hbr = torch.tensor(binary_hbr_weights, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            cnn = Conv2d(1, 1, kernel_size=(1, 4), stride=(1, 4), padding=(0, 1), padding_mode='replicate')
            cnn.weight.data.fill_(1.0 / 4.0)
            cnn.bias.data.fill_(0.0)
            
            convolved_hbo = cnn(tensor_hbo).squeeze(0).squeeze(0).detach().numpy()
            convolved_hbr = cnn(tensor_hbr).squeeze(0).squeeze(0).detach().numpy()
            
            # Sum across time to get channel importance
            summed_hbo_ch = np.sum(convolved_hbo, axis=1)
            summed_hbr_ch = np.sum(convolved_hbr, axis=1)
            
            count_channels += summed_hbo_ch
            count_channels_hbr += summed_hbr_ch
        
        # Aggregate by brain area
        for brain_area, channel_ids in self.brain_areas_mapping.items():
            area_importance = []
            
            for ch_id in channel_ids:
                if ch_id in hbo_names_o:
                    idx = hbo_names_o.index(ch_id)
                    # Combine HbO and HbR for this channel
                    importance = (count_channels[idx] + count_channels_hbr[idx]) / 2
                    area_importance.append(importance)
            
            if len(area_importance) > 0:
                area_data[brain_area] = np.array(area_importance)
            else:
                area_data[brain_area] = np.array([0])
        
        return area_data
    
    def analyze_subject(self, subject, model='svc'):
        """
        Perform statistical analysis for a single subject.
        
        Parameters:
        -----------
        subject : int
            Subject ID
        model : str
            'svc' or 'lda'
            
        Returns:
        --------
        results : dict
            Statistical results for this subject
        """
        
        print(f"\n{'='*70}")
        print(f"SUBJECT {subject} - STATISTICAL ANALYSIS")
        print(f"{'='*70}")
        
        # Load and aggregate data
        area_data = self.load_shap_and_aggregate(subject, model)
        
        if area_data is None:
            return None
        
        # Aggregate each area to single value (mean across channels)
        area_means = {area: np.mean(values) for area, values in area_data.items()}
        
        print("\nBrain Area Importance (Mean SHAP):")
        for area, value in sorted(area_means.items(), key=lambda x: x[1], reverse=True):
            print(f"  {area:10s}: {value:.6f}")
        
        # Prepare data for statistical tests
        test_data = {area: values for area, values in area_data.items() if len(values) > 0}
        
        # 1. Descriptive statistics
        print("\n" + "-"*70)
        print("DESCRIPTIVE STATISTICS")
        print("-"*70)
        
        desc_stats = []
        for area in sorted(test_data.keys()):
            values = test_data[area]
            desc_stats.append({
                'Area': area,
                'N_channels': len(values),
                'Mean': np.mean(values),
                'SD': np.std(values, ddof=1) if len(values) > 1 else 0,
                'Median': np.median(values),
                'Min': np.min(values),
                'Max': np.max(values)
            })
        
        desc_df = pd.DataFrame(desc_stats)
        print(desc_df.to_string(index=False))
        
        # 2. Check normality
        print("\n" + "-"*70)
        print("NORMALITY TEST (Shapiro-Wilk)")
        print("-"*70)
        
        normality_results = {}
        for area, values in test_data.items():
            if len(values) >= 3:
                w_stat, p_value = shapiro(values)
                is_normal = p_value > self.alpha
                normality_results[area] = {'w_stat': w_stat, 'p_value': p_value, 'is_normal': is_normal}
                status = "Normal" if is_normal else "Non-normal"
                print(f"{area:10s}: W={w_stat:.4f}, p={p_value:.4f} ({status})")
            else:
                normality_results[area] = {'w_stat': None, 'p_value': None, 'is_normal': None}
                print(f"{area:10s}: N/A (less than 3 samples)")
        
        # 3. Overall test (Kruskal-Wallis)
        print("\n" + "-"*70)
        print("KRUSKAL-WALLIS TEST (Overall difference)")
        print("-"*70)
        
        data_arrays = [test_data[area] for area in sorted(test_data.keys())]
        h_stat, p_kruskal = kruskal(*data_arrays)
        
        print(f"H statistic: {h_stat:.4f}")
        print(f"p-value: {p_kruskal:.6f}")
        print(f"Significant: {'YES' if p_kruskal < self.alpha else 'NO'}")
        
        # 4. Pairwise comparisons
        print("\n" + "-"*70)
        print("PAIRWISE COMPARISONS (Mann-Whitney U)")
        print("-"*70)
        
        pairwise_results = []
        areas = sorted(test_data.keys())
        
        for i in range(len(areas)):
            for j in range(i + 1, len(areas)):
                area1, area2 = areas[i], areas[j]
                data1 = test_data[area1]
                data2 = test_data[area2]
                
                # Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Effect size
                n = len(data1) + len(data2)
                z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 0
                effect_size = z_score / np.sqrt(n) if n > 0 else 0
                
                pairwise_results.append({
                    'Area_1': area1,
                    'Area_2': area2,
                    'Mean_1': np.mean(data1),
                    'Mean_2': np.mean(data2),
                    'Difference': np.mean(data1) - np.mean(data2),
                    'U_statistic': u_stat,
                    'p_value': p_value,
                    'Effect_size_r': effect_size
                })
        
        pairwise_df = pd.DataFrame(pairwise_results)
        
        # Apply FDR correction
        if len(pairwise_df) > 0:
            _, p_corrected, _, _ = multipletests(
                pairwise_df['p_value'],
                alpha=self.alpha,
                method='fdr_bh'
            )
            pairwise_df['p_corrected_FDR'] = p_corrected
            pairwise_df['Significant_raw'] = pairwise_df['p_value'] < self.alpha
            pairwise_df['Significant_FDR'] = pairwise_df['p_corrected_FDR'] < self.alpha
            
            # Sort by p-value
            pairwise_df = pairwise_df.sort_values('p_value')
            
            print("\nTop 10 significant comparisons:")
            display_df = pairwise_df[['Area_1', 'Area_2', 'Difference', 'p_value', 'p_corrected_FDR', 'Effect_size_r']].head(10)
            print(display_df.to_string(index=False))
            
            sig_pairs = pairwise_df[pairwise_df['Significant_FDR']]
            if len(sig_pairs) > 0:
                print(f"\nSignificant pairs (FDR corrected): {len(sig_pairs)}")
                for _, row in sig_pairs.iterrows():
                    print(f"  {row['Area_1']} vs {row['Area_2']}: "
                          f"diff={row['Difference']:.4f}, p_FDR={row['p_corrected_FDR']:.6f}")
            else:
                print("\nNo significant pairs after FDR correction")
        
        # Save results
        self.save_subject_results(subject, desc_df, pairwise_df, area_means, model)
        
        # Create visualizations
        self.create_subject_visualizations(subject, test_data, area_means, pairwise_df, h_stat, p_kruskal, model)
        
        return {
            'area_data': area_data,
            'area_means': area_means,
            'descriptive': desc_df,
            'pairwise': pairwise_df,
            'kruskal': {'h_stat': h_stat, 'p_value': p_kruskal}
        }
    
    def save_subject_results(self, subject, desc_df, pairwise_df, area_means, model):
        """Save results for a subject."""
        
        subject_folder = os.path.join(self.output_folder, f'subject_{subject}')
        os.makedirs(subject_folder, exist_ok=True)
        
        # Descriptive statistics
        desc_df.to_csv(os.path.join(subject_folder, 'descriptive_statistics.csv'), index=False)
        
        # Pairwise comparisons
        if len(pairwise_df) > 0:
            pairwise_df.to_csv(os.path.join(subject_folder, 'pairwise_comparisons.csv'), index=False)
        
        # Area importance summary
        area_summary = pd.DataFrame([
            {'Area': area, 'Mean_Importance': value}
            for area, value in sorted(area_means.items(), key=lambda x: x[1], reverse=True)
        ])
        area_summary.to_csv(os.path.join(subject_folder, 'area_importance_ranking.csv'), index=False)
    
    def create_subject_visualizations(self, subject, test_data, area_means, pairwise_df, h_stat, p_kruskal, model):
        """Create visualizations for a subject."""
        
        subject_folder = os.path.join(self.output_folder, f'subject_{subject}')
        os.makedirs(subject_folder, exist_ok=True)
        
        # 1. Box plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        areas = sorted(test_data.keys())
        box_data = [test_data[area] for area in areas]
        
        bp = ax.boxplot(box_data, labels=areas, patch_artist=True, widths=0.6)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(areas)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, area in enumerate(areas):
            y = test_data[area]
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=80, color='black', zorder=3)
        
        ax.set_ylabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
        ax.set_xlabel('Brain Area', fontsize=12, fontweight='bold')
        ax.set_title(f'Subject {subject} - SHAP Importance by Brain Area\n' +
                     f'Kruskal-Wallis: H={h_stat:.2f}, p={p_kruskal:.4f}',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_folder, f'boxplot_areas_{model}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot with ranking
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_areas = sorted(area_means.keys(), key=lambda x: area_means[x], reverse=True)
        sorted_values = [area_means[area] for area in sorted_areas]
        
        bars = ax.barh(range(len(sorted_areas)), sorted_values, color=colors[:len(sorted_areas)], 
                        edgecolor='black', linewidth=1.5, alpha=0.7)
        
        ax.set_yticks(range(len(sorted_areas)))
        ax.set_yticklabels(sorted_areas)
        ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Subject {subject} - Brain Area Importance Ranking', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            ax.text(value, i, f' {value:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_folder, f'ranking_areas_{model}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of p-values (if enough comparisons)
        if len(pairwise_df) > 0:
            n_areas = len(areas)
            p_matrix = np.ones((n_areas, n_areas))
            
            for _, row in pairwise_df.iterrows():
                i = areas.index(row['Area_1'])
                j = areas.index(row['Area_2'])
                p_matrix[i, j] = row['p_corrected_FDR']
                p_matrix[j, i] = row['p_corrected_FDR']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            log_p_matrix = -np.log10(p_matrix + 1e-10)
            
            im = ax.imshow(log_p_matrix, cmap='RdYlGn', aspect='auto')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('-log₁₀(p-value FDR)', fontsize=11)
            
            ax.set_xticks(range(n_areas))
            ax.set_yticks(range(n_areas))
            ax.set_xticklabels(areas, rotation=45, ha='right')
            ax.set_yticklabels(areas)
            
            # Add text annotations
            for i in range(n_areas):
                for j in range(n_areas):
                    if i != j:
                        text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(f'Subject {subject} - Pairwise P-values (FDR corrected)',
                        fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(subject_folder, f'pvalue_heatmap_{model}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_all_subjects(self, subjects, model='svc'):
        """
        Analyze all subjects.
        """
        
        print("\n" + "#"*70)
        print("# PER-SUBJECT STATISTICAL ANALYSIS OF SHAP VALUES")
        print("#"*70)
        
        all_results = {}
        
        for subject in subjects:
            results = self.analyze_subject(subject, model)
            if results is not None:
                all_results[subject] = results
        
        # Create summary across subjects
        self.create_summary_report(all_results, model)
        
        print("\n" + "#"*70)
        print("# ANALYSIS COMPLETE")
        print("#"*70)
        print(f"\nResults saved in: {self.output_folder}/")
    
    def create_summary_report(self, all_results, model):
        """Create a summary report across all subjects."""
        
        output_file = os.path.join(self.output_folder, f'summary_report_all_subjects_{model}.txt')
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PER-SUBJECT STATISTICAL ANALYSIS SUMMARY\n")
            f.write(f"Model: {model.upper()}\n")
            f.write(f"Significance level: α = {self.alpha}\n")
            f.write("="*70 + "\n\n")
            
            # For each subject, write summary
            for subject in sorted(all_results.keys()):
                results = all_results[subject]
                
                f.write(f"\nSUBJECT {subject}\n")
                f.write("-"*70 + "\n")
                
                f.write("\nBrain Area Importance Ranking:\n")
                ranked = sorted(results['area_means'].items(), key=lambda x: x[1], reverse=True)
                for i, (area, value) in enumerate(ranked, 1):
                    f.write(f"  {i}. {area:10s}: {value:.6f}\n")
                
                f.write(f"\nOverall Test (Kruskal-Wallis):\n")
                f.write(f"  H = {results['kruskal']['h_stat']:.4f}\n")
                f.write(f"  p = {results['kruskal']['p_value']:.6f}\n")
                f.write(f"  Significant: {'YES' if results['kruskal']['p_value'] < self.alpha else 'NO'}\n")
                
                # Significant pairs
                if len(results['pairwise']) > 0:
                    sig_pairs = results['pairwise'][results['pairwise']['Significant_FDR']]
                    if len(sig_pairs) > 0:
                        f.write(f"\nSignificant Pairs (FDR corrected): {len(sig_pairs)}\n")
                        for _, row in sig_pairs.iterrows():
                            f.write(f"  {row['Area_1']} vs {row['Area_2']}: "
                                  f"p_FDR={row['p_corrected_FDR']:.6f}\n")
                    else:
                        f.write("\nNo significant pairs (FDR corrected)\n")
                
                f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['svc', 'lda'], default='svc')
    parser.add_argument('--shap-root', default='outputs/shap')
    parser.add_argument('--output-folder', default='outputs/shap_analysis')
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    subjects = [0, 1, 2, 3, 4, 5, 6, 7]
    analyzer = PerSubjectShapAnalysis(
        shap_folder=args.shap_root,
        output_folder=args.output_folder,
        alpha=args.alpha,
    )
    analyzer.analyze_all_subjects(subjects, model=args.model)


if __name__ == '__main__':
    main()
