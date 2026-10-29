"""Evaluation utilities: classification metrics, stratified k-fold
cross-validation and decision-threshold tuning.

The data is heavily imbalanced (~8.8% positives), so the metric of interest
is the F1 score of the positive class, not accuracy.
"""

import numpy as np


def classification_metrics(y_true, y_pred):
    """Metrics for labels in {0, 1}. Returns a dict with f1, precision,
    recall and accuracy."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": np.mean(y_pred == y_true),
    }


def best_threshold(y_true, scores):
    """Threshold on `scores` that maximizes F1 for labels in {0, 1}.

    Sorting the scores once lets us evaluate every possible cut in O(N log N):
    predicting the top-k scores as positive gives tp = cumsum of labels.
    """
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    f1 = 2 * tp / (k + y_true.sum())
    # Only cut between distinct score values.
    s_sorted = scores[order]
    valid = np.append(s_sorted[1:] < s_sorted[:-1], True)
    i = np.argmax(np.where(valid, f1, -1))
    return s_sorted[i], f1[i]


def stratified_kfold(y, k_fold, seed=1):
    """Split indices into k folds that each keep the class proportions.

    Returns a list of (train_idx, val_idx) pairs.
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k_fold)]
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        for f, part in enumerate(np.array_split(idx, k_fold)):
            folds[f].append(part)
    folds = [np.concatenate(f) for f in folds]
    return [
        (np.concatenate(folds[:f] + folds[f + 1 :]), folds[f]) for f in range(k_fold)
    ]


def cross_validate(
    x, y, columns, make_preprocessor, fit, k_fold=5, seed=1, verbose=True
):
    """Stratified k-fold cross-validation of a full pipeline.

    For every fold, a fresh preprocessor is fitted on the training part only,
    the model is trained, and validation scores are stored. The decision
    threshold is then chosen to maximize F1 on all out-of-fold scores, and
    per-fold metrics are reported at that threshold.

    Note: choosing one threshold on the pooled out-of-fold scores uses the
    validation labels for a single scalar, so the estimate is very slightly
    optimistic; with ~65k validation points per fold the effect is tiny.

    Args:
        x: cleaned raw features (output of clean_raw), shape (N, D).
        y: labels in {0, 1}, shape (N,).
        columns: names of the columns of x.
        make_preprocessor: function () -> unfitted Preprocessor.
        fit: function (tx, y) -> score_fn, where score_fn(tx) returns a
            real-valued score per row (higher = more likely positive).

    Returns:
        dict with "threshold", "oof_scores", per-fold metric lists and their
        means/stds.
    """
    oof = np.empty(len(y))
    splits = stratified_kfold(y, k_fold, seed)
    for f, (tr, va) in enumerate(splits):
        prep = make_preprocessor().fit(x[tr], columns)
        score_fn = fit(prep.transform(x[tr]), y[tr])
        oof[va] = score_fn(prep.transform(x[va]))
        if verbose:
            print(f"  fold {f + 1}/{k_fold} done")

    threshold, _ = best_threshold(y, oof)
    per_fold = [
        classification_metrics(y[va], (oof[va] >= threshold).astype(int))
        for _, va in splits
    ]
    result = {"threshold": threshold, "oof_scores": oof}
    for key in per_fold[0]:
        vals = np.array([m[key] for m in per_fold])
        result[key] = vals
        result[key + "_mean"] = vals.mean()
        result[key + "_std"] = vals.std()
    return result
