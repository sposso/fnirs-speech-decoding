"""
Nested cross-validation training for linear SVC and LDA, with SHAP value extraction.

This is the core training loop used to produce:
  - Table I (decoding AUC per subject)
  - SHAP value arrays consumed by the figure scripts (Fig 1-3) and Table II.

Pipeline per outer fold:
  1. Stratified split (5 outer folds, seed=22).
  2. Per-fold min-max normalization fit on train only.
  3. ADASYN oversampling of the minority class on the training set.
  4. Inner 3-fold grid search on C (SVC only); LDA has no hyperparameters here.
  5. Predict on the test fold; record AUC.
  6. Fit a SHAP LinearExplainer on the training fold and compute SHAP values
     on the test fold; multiply by the test labels (+/-1) so that positive SHAP
     always indicates "supports the correct class" (paper Section II.D).
"""
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import ADASYN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


OUTER_K = 5
INNER_K = 3
SEED = 22
MAX_ITER = 250_000
C_LIST = [1e-3, 1e-2, 1e-1, 1.0]


def machine_learning_exp(model, X, Y):
    """Run nested CV for one (model, subject) and return SHAP values + AUCs.

    Args:
        model: 'svc' or 'lda'.
        X: pd.DataFrame of shape (n_trials, n_features). Features are flattened
           (n_channels * n_times) so SHAP can attribute to each (channel, time).
        Y: np.ndarray of shape (n_trials,) with values in {-1, +1}.

    Returns:
        all_shap_values: list of length OUTER_K, each (n_test, n_features).
        all_hps: list of best hyperparameters per fold (None for LDA).
        auc: list of OUTER_K AUC scores on each held-out test fold.
        all_y_true: concatenated test labels across folds.
    """
    if model not in ("svc", "lda"):
        raise ValueError(f"unsupported model: {model}")

    out_kf = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=SEED)
    in_kf = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=SEED)

    all_shap_values, all_hps, auc, all_y_true = [], [], [], []

    for k, (train_idx, test_idx) in enumerate(out_kf.split(X, Y)):
        print(f"  Outer fold {k + 1}/{OUTER_K}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
        all_y_true.extend(Y_test)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

        # Per-fold min-max normalization (fit on train only)
        mins = X_train.min(axis=0)
        maxs = X_train.max(axis=0)
        denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
        X_train = (X_train - mins) / denom
        X_test = (X_test - mins) / denom

        # ADASYN class balancing on training fold only
        X_train, Y_train = ADASYN(random_state=SEED).fit_resample(X_train, Y_train)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        Y_train = pd.Series(Y_train)

        if model == "lda":
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, Y_train)
            best_estimator = clf
            all_hps.append(None)
        else:  # svc
            in_split = in_kf.split(X_train, Y_train)
            grid = GridSearchCV(
                LinearSVC(max_iter=MAX_ITER, dual="auto"),
                {"C": C_LIST},
                scoring="accuracy",
                cv=in_split,
            )
            grid.fit(X_train, Y_train)
            best_estimator = grid.best_estimator_
            all_hps.append(grid.best_params_["C"])

        y_pred = best_estimator.predict(X_test)
        auc.append(roc_auc_score(Y_test, y_pred))

        # SHAP values on the held-out test fold
        explainer = shap.LinearExplainer(best_estimator, X_train)
        shap_values = explainer.shap_values(X_test)
        # Sign-correct so positive SHAP always supports the correct class
        shap_values = shap_values * Y_test.reshape(-1, 1)
        all_shap_values.append(shap_values)

    return all_shap_values, all_hps, auc, all_y_true
