"""
LOSO classification loop with AUC-ROC, bootstrap CIs, and DeLong test.

Feature sets:
  MS    — 26 microstate features
  SP    — 21 spectral features
  MS+SP — 47 combined features

Classifiers (per feature set):
  LDA — shrinkage LDA  (primary)
  SVM — RBF-SVM, C tuned via nested 5-fold CV on training subjects
  LR  — Logistic Regression

Primary metric: AUC-ROC (threshold-independent, robust to class imbalance).
Secondary metrics at Youden's J threshold:
  balanced accuracy, sensitivity, specificity, F1.

Confidence intervals: 10,000-resample bootstrap.
Statistical comparisons: DeLong (1988) paired AUC test, Bonferroni α = 0.017.
"""

import numpy as np
from scipy.stats import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


# ── Classifier factory ─────────────────────────────────────────────────────────

def _make_classifiers():
    """Return dict of {label: classifier_instance}."""
    return {
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "SVM": SVC(kernel="rbf", probability=True),   # C tuned per fold
        "LR":  LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000),
    }


def _tune_svm_C(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Select RBF-SVM C via nested 5-fold CV on training subjects."""
    C_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    if len(y_train) < 10:
        return 1.0   # too few samples to do inner CV

    best_C, best_auc = 1.0, -1.0
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for C in C_candidates:
        aucs = []
        for tr, va in skf.split(X_train, y_train):
            clf = SVC(kernel="rbf", C=C, probability=True)
            clf.fit(X_train[tr], y_train[tr])
            p = clf.predict_proba(X_train[va])[:, 1]
            if len(np.unique(y_train[va])) == 2:
                aucs.append(roc_auc_score(y_train[va], p))
        if aucs and np.mean(aucs) > best_auc:
            best_auc = np.mean(aucs)
            best_C = C
    return best_C


# ── LOSO loop ──────────────────────────────────────────────────────────────────

def loso_cv_lda_fast(X: np.ndarray, y: np.ndarray) -> float:
    """
    LDA-only LOSO cross-validation returning AUC.
    No SVM/LR overhead — designed for use inside permutation loops.

    Parameters
    ----------
    X : (n_subjects, n_features)
    y : (n_subjects,) binary labels

    Returns
    -------
    float — AUC-ROC
    """
    n = len(y)
    y_true_all = []
    y_prob_all = []

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[[i]],       y[[i]]

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_train)
        X_te_s   = scaler.transform(X_test)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_tr_s, y_train)
        prob = lda.predict_proba(X_te_s)[0, 1]

        y_true_all.append(int(y_test[0]))
        y_prob_all.append(float(prob))

    yt = np.array(y_true_all)
    yp = np.array(y_prob_all)
    if len(np.unique(yt)) < 2:
        return 0.5
    return float(roc_auc_score(yt, yp))


def loso_cv(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Leave-one-subject-out cross-validation for all classifiers.

    Parameters
    ----------
    X : (n_subjects, n_features)
    y : (n_subjects,) binary labels

    Returns
    -------
    dict: {clf_label: {'y_true': [...], 'y_prob': [...]}}
    """
    n = len(y)
    results = {clf: {"y_true": [], "y_prob": []} for clf in ["LDA", "SVM", "LR"]}

    for i in range(n):
        # Train/test split
        train_idx = [j for j in range(n) if j != i]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[[i]], y[[i]]

        # Standardise using training statistics only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Fit and predict with each classifier
        for clf_label, clf in _make_classifiers().items():
            # For SVM, tune C on training subjects
            if clf_label == "SVM":
                best_C = _tune_svm_C(X_train_s, y_train)
                clf = SVC(kernel="rbf", C=best_C, probability=True)

            clf.fit(X_train_s, y_train)
            prob = clf.predict_proba(X_test_s)[0, 1]   # probability of class 1 (viable)
            results[clf_label]["y_true"].append(int(y_test[0]))
            results[clf_label]["y_prob"].append(float(prob))

    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute AUC, balanced accuracy, sensitivity, specificity, F1
    at the Youden's J optimal threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auc = float(roc_auc_score(y_true, y_prob))

    # Youden's J: sensitivity + specificity - 1, maximised over thresholds
    from sklearn.metrics import roc_curve, f1_score
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])

    y_pred = (y_prob >= best_thresh).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc     = (sensitivity + specificity) / 2.0
    f1          = f1_score(y_true, y_pred, zero_division=0)

    return {
        "auc":          auc,
        "bal_acc":      float(bal_acc),
        "sensitivity":  float(sensitivity),
        "specificity":  float(specificity),
        "f1":           float(f1),
        "threshold":    best_thresh,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple:
    """
    Percentile bootstrap confidence interval for AUC.

    Returns
    -------
    (lower, upper) — two-sided (1-alpha) CI
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    auc_boot = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            auc_boot[b] = 0.5     # degenerate sample
        else:
            auc_boot[b] = roc_auc_score(yt, yp)

    lo = float(np.percentile(auc_boot, 100 * alpha / 2))
    hi = float(np.percentile(auc_boot, 100 * (1 - alpha / 2)))
    return lo, hi


# ── DeLong test ────────────────────────────────────────────────────────────────
# Nonparametric paired AUC comparison from DeLong et al. (1988).
# Implementation based on the structural components (V statistics) approach.

def _delong_structural_components(y_true, y_prob):
    """Compute V10 and V01 structural components for DeLong's method."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = y_prob[y_true == 1]   # probabilities for positive class
    neg = y_prob[y_true == 0]   # probabilities for negative class
    m, n = len(pos), len(neg)

    # V10[i]: for each positive, how it compares to all negatives
    V10 = np.zeros(m)
    for i, p in enumerate(pos):
        V10[i] = np.mean((neg < p).astype(float) + 0.5 * (neg == p).astype(float))

    # V01[j]: for each negative, how it compares to all positives
    V01 = np.zeros(n)
    for j, q in enumerate(neg):
        V01[j] = np.mean((pos > q).astype(float) + 0.5 * (pos == q).astype(float))

    auc = float(V10.mean())
    return auc, V10, V01, m, n


def delong_test(y_true, y_prob_a, y_prob_b) -> dict:
    """
    DeLong (1988) nonparametric paired test comparing two AUC values
    computed from the same set of subjects.

    Returns
    -------
    dict with: auc_a, auc_b, z, p_value
    """
    auc_a, V10_a, V01_a, m, n = _delong_structural_components(y_true, y_prob_a)
    auc_b, V10_b, V01_b, _, _ = _delong_structural_components(y_true, y_prob_b)

    # Covariance matrix of (AUC_a, AUC_b) — 2x2
    S10_aa = float(np.var(V10_a, ddof=1)) / m
    S10_bb = float(np.var(V10_b, ddof=1)) / m
    S10_ab = float(np.cov(V10_a, V10_b, ddof=1)[0, 1]) / m

    S01_aa = float(np.var(V01_a, ddof=1)) / n
    S01_bb = float(np.var(V01_b, ddof=1)) / n
    S01_ab = float(np.cov(V01_a, V01_b, ddof=1)[0, 1]) / n

    var_diff = (S10_aa + S01_aa) + (S10_bb + S01_bb) - 2 * (S10_ab + S01_ab)

    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z": 0.0, "p_value": 1.0}

    z       = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = float(1.0 - chi2.cdf(z**2, df=1))   # two-sided: chi2(1) = Z^2, captures both tails

    return {
        "auc_a":   float(auc_a),
        "auc_b":   float(auc_b),
        "z":       float(z),
        "p_value": p_value,
    }


# ── Feature importance ─────────────────────────────────────────────────────────

def lda_feature_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit LDA on all subjects; return |coefficient| rescaled to [0, 1].
    Post-hoc only — does not affect any reported AUC.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_s, y)
    coef = np.abs(lda.coef_[0])
    coef_scaled = (coef - coef.min()) / (coef.max() - coef.min() + 1e-30)
    return coef_scaled


def spearman_correlations(X: np.ndarray, fold3_acc: np.ndarray) -> dict:
    """
    Spearman correlation between each feature and continuous Fold-3 accuracy.
    Bonferroni correction: alpha / n_features.

    Returns
    -------
    dict with: rho (n_features,), p_values (n_features,), corrected_alpha
    """
    from scipy.stats import spearmanr
    n_features = X.shape[1]
    rho     = np.zeros(n_features)
    p_vals  = np.zeros(n_features)

    for j in range(n_features):
        r, p = spearmanr(X[:, j], fold3_acc)
        rho[j]    = float(r)
        p_vals[j] = float(p)

    corrected_alpha = 0.05 / n_features

    return {
        "rho":             rho.tolist(),
        "p_values":        p_vals.tolist(),
        "corrected_alpha": corrected_alpha,
        "significant_idx": [int(i) for i in np.where(p_vals < corrected_alpha)[0]],
    }
