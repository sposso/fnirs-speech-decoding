# Factors Influencing Speech Perception Decoding with fNIRS

Code to reproduce the results of:

> **Factors Influencing Speech Perception Decoding with fNIRS: Analysis of Spatial and Temporal Characteristics and Signal Complexity**
> Santiago Posso-Murillo, Luis G. Sanchez-Giraldo, Jihye Bae
> *University of Kentucky, Department of Electrical Engineering*

We decode speech perception versus silence from fNIRS recordings of 8 healthy
adults using **Linear SVC** and **LDA**, then probe the trained decoders with
**SHAP** to identify which brain regions and time points carry the speech-
relevant information. We additionally compute **Modified Multiscale Entropy
(MME)** to relate per-subject signal complexity to decoding accuracy.

## Repository contents

| File | Produces |
|---|---|
| `audio_data.py` | Preprocessing pipeline (Beer-Lambert, TDDR, short-channel regression, bandpass, epoching). |
| `train_shap.py` | Nested 5×3 cross-validation training of SVC / LDA, plus SHAP value extraction. |
| `run_decoding.py` | AUC table and SHAP values. |
| `notebooks/01_decoding_table.ipynb` | Renders **Table I**. |
| `fig_1_and_3.py` | **Fig. 1** (subject-wise SHAP) and **Fig. 3** (group-averaged SHAP), per model. |
| `fig_2.py` | **Fig. 2** (brain montage of SHAP relevance), per model. |
| `fig_4.py` | **Fig. 4** (MME complexity per subject + PC1 score by performance group). |

## Dataset

We use the public dataset from Shader et al. [^shader].

[^shader]: Shader et al., *Hearing Research*, 2018. Auditory-only and visual-only speech in fNIRS, BIDS-formatted, 8 adults, 18 trials per condition.

## Setup

```bash
git clone <this-repo>
cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```


## How to reproduce the paper

### Step 1 — Train the decoders (Table I)

```bash
python run_decoding.py
```

This loops over 8 subjects × {SVC, LDA} with the nested cross-validation
described in Section II.D of the paper. Outputs:

```
outputs/
├── auc/
│   └── auc_per_subject.csv               
└── shap/
    └── subject_{0..7}/
        ├── shap_values_svc_subject_*.npz
        └── shap_values_lda_subject_*.npz  # 5 outer folds per file
```

For the table-formatted version of Table I, run the notebook:

```bash
jupyter notebook notebooks/01_decoding_table.ipynb
```

The notebook re-runs the same training pipeline and renders Table I.

### Step 2 — Generate Figures 1–4

```bash
python fig_1_and_3.py --model svc        # Fig. 1a, 1c, 3a, 3c (paper)
python fig_1_and_3.py --model lda        # Fig. 1b, 1d, 3b, 3d (paper)
python fig_2.py   --model svc        # Fig. 2 (top SVC, paper)
python fig_2.py   --model lda        # Fig. 2 (LDA variant)
python fig_4.py                      # Fig. 4a + 4b
```

Outputs land in `figures/`. Figure 4 caches its MME computation in
`outputs/mme/entropy_df.pkl`; delete that file to force recomputation.


Per-subject area rankings, Kruskal-Wallis omnibus tests, and pairwise
Mann-Whitney U tests with effect sizes land in
`outputs/shap_analysis/subject_*/`.

## Method summary

| Stage | Choice | Rationale |
|---|---|---|
| Preprocessing | optical density → SCI < 0.8 channel rejection → TDDR motion correction → short-channel regression → Beer-Lambert (HbO/HbR) → 0.02–0.4 Hz bandpass → negative-correlation enhancement | Same as Shader et al. [18] in the paper. |
| Epoching | 0–18 s post-stimulus, 100 µM amplitude rejection | Captures the 12.5 s audio plus ≈6 s hemodynamic peak. |
| Class balancing | ADASYN on training fold only | Auditory (18 trials) vs silence (10 trials) imbalance. |
| Cross-validation | 5 outer × 3 inner stratified folds | Outer estimates AUC; inner tunes SVC's `C`. |
| Decoders | Linear SVC (`C ∈ {1e-3, 1e-2, 1e-1, 1}`), LDA (SVD solver) | High-dimensional fNIRS data, low-data regime. |
| Metric | AUC | Robust to class imbalance. |
| Interpretation | `shap.LinearExplainer` per fold, sign-corrected by test labels | Positive SHAP always supports the correct class. |
| Complexity | Modified Multiscale Entropy with embedding m=2, tolerance r=0.2·σ, scales 1–20 | Quantifies coarse-grained irregularity per subject. |

## Citation

```bibtex
@inproceedings{possomurillo2025fnirs,
  title  = {Factors Influencing Speech Perception Decoding with fNIRS:
            Analysis of Spatial and Temporal Characteristics and Signal Complexity},
  author = {Posso-Murillo, Santiago and Sanchez-Giraldo, Luis G. and Bae, Jihye},
  year   = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

Santiago Posso-Murillo — `spo230@uky.edu`
