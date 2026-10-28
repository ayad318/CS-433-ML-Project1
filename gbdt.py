"""Gradient-boosted decision trees for binary classification, in NumPy.

A small histogram-based implementation of gradient boosting on the logistic
loss (the same idea as XGBoost / LightGBM, written from scratch since no ML
library is allowed):

  * every feature is discretized once into at most 32 bins (quantiles of the
    training data; the last bin holds missing values), so a split search only
    needs per-bin sums of gradients and Hessians, computed with np.bincount;
  * each tree is grown level by level to a fixed depth; a node is split at
    the (feature, bin) with the largest second-order gain
        G_L^2 / (H_L + lam) + G_R^2 / (H_R + lam) - G^2 / (H + lam);
  * leaves predict the Newton step -G / (H + lam), scaled by the learning rate;
  * each tree sees a random half of the rows and of the features, which
    regularizes the model and halves the training time.

Trees work on the raw survey codes directly, so they need none of the
one-hot encoding or scaling done for logistic regression, and they pick up
interactions between answers (e.g. age x diabetes) on their own.
"""

import numpy as np

from helpers import sigmoid

N_BINS = 32  # bin N_BINS - 1 is reserved for missing values


def fit_bins(x):
    """Compute bin edges for every column of x from the training data.

    Columns with few distinct values get one bin per value; the others get
    quantile bins. NaNs are ignored here and go to the missing-value bin.

    Args:
        x: numpy array of shape (N, F), raw features with NaNs.

    Returns:
        list of F sorted arrays of bin edges.
    """
    edges = []
    for f in range(x.shape[1]):
        col = x[:, f]
        values = np.unique(col[~np.isnan(col)])
        if len(values) <= N_BINS - 1:
            # midpoints between consecutive distinct values
            edges.append((values[:-1] + values[1:]) / 2)
        else:
            quantiles = np.linspace(0, 1, N_BINS)[1:-1]
            edges.append(np.unique(np.quantile(col[~np.isnan(col)], quantiles)))
    return edges


def apply_bins(x, edges):
    """Map raw features to bin indices in [0, N_BINS - 1] (uint8)."""
    out = np.empty(x.shape, np.uint8)
    for f, e in enumerate(edges):
        b = np.searchsorted(e, x[:, f], side="right")
        b[np.isnan(x[:, f])] = N_BINS - 1
        out[:, f] = b
    return out


def build_tree(xb, g, h, depth, lam, min_hessian, features):
    """Grow one regression tree on gradients g and Hessians h.

    Nodes are numbered as in a binary heap: the root is 0 and node i has
    children 2i + 1 (bin <= threshold) and 2i + 2 (bin > threshold). Samples
    reaching a node that does not split are all sent to its left child, so
    every sample ends in one of the 2^depth leaves.

    Args:
        xb: uint8 array of shape (N, F), binned features.
        g, h: arrays of shape (N,), gradient and Hessian of the loss.
        depth: int, depth of the tree.
        lam: float, L2 penalty on leaf values.
        min_hessian: float, minimum Hessian sum in each child of a split.
        features: boolean array of shape (F,), the features this tree may use.

    Returns:
        splits: dict {node id: (feature, threshold bin)}.
        leaves: array of shape (2^depth,), the leaf values -G / (H + lam).
    """
    n = xb.shape[0]
    node = np.zeros(n, np.int64)
    splits = {}
    for d in range(depth):
        n_nodes = 2**d
        local = node - (n_nodes - 1)  # index of each sample's node in this level
        g_tot = np.bincount(local, g, n_nodes)
        h_tot = np.bincount(local, h, n_nodes)
        parent_score = g_tot**2 / (h_tot + lam)
        best_gain = np.zeros(n_nodes)
        best_feature = np.full(n_nodes, -1)
        best_bin = np.zeros(n_nodes, np.int64)
        for f in np.flatnonzero(features):
            # histogram of g and h per (node, bin) in a single bincount
            idx = local * N_BINS + xb[:, f]
            g_hist = np.bincount(idx, g, n_nodes * N_BINS).reshape(n_nodes, N_BINS)
            h_hist = np.bincount(idx, h, n_nodes * N_BINS).reshape(n_nodes, N_BINS)
            # left child = bins <= threshold, for every threshold at once
            g_left = np.cumsum(g_hist, axis=1)[:, :-1]
            h_left = np.cumsum(h_hist, axis=1)[:, :-1]
            g_right = g_tot[:, None] - g_left
            h_right = h_tot[:, None] - h_left
            gain = (
                g_left**2 / (h_left + lam)
                + g_right**2 / (h_right + lam)
                - parent_score[:, None]
            )
            gain[(h_left < min_hessian) | (h_right < min_hessian)] = 0
            k = gain.argmax(axis=1)
            v = gain[np.arange(n_nodes), k]
            better = v > best_gain
            best_gain[better] = v[better]
            best_feature[better] = f
            best_bin[better] = k[better]

        # Record the chosen splits; nodes where no split has positive gain
        # (or none respects min_hessian) stay unsplit.
        for j in np.flatnonzero(best_feature >= 0):
            splits[n_nodes - 1 + j] = (best_feature[j], best_bin[j])
        # Send each sample to its child for the next level.
        f_of_sample = best_feature[local]
        go_right = np.zeros(n, bool)
        has_split = f_of_sample >= 0
        go_right[has_split] = (
            xb[has_split, f_of_sample[has_split]] > best_bin[local[has_split]]
        )
        node = 2 * node + 1 + go_right

    # Leaf value = Newton step on the logistic loss for the samples in it.
    n_leaves = 2**depth
    local = node - (n_leaves - 1)
    leaves = -np.bincount(local, g, n_leaves) / (np.bincount(local, h, n_leaves) + lam)
    return splits, leaves


def predict_tree(xb, splits, leaves, depth):
    """Return the leaf value reached by every row of the binned features xb."""
    n = xb.shape[0]
    node = np.zeros(n, np.int64)
    for _ in range(depth):
        # Look up, for every sample, the split stored at its current node.
        feature = np.full(n, -1)
        threshold = np.zeros(n, np.int64)
        for node_id, (f, b) in splits.items():
            here = node == node_id
            feature[here] = f
            threshold[here] = b
        go_right = np.zeros(n, bool)
        has_split = feature >= 0
        go_right[has_split] = xb[has_split, feature[has_split]] > threshold[has_split]
        node = 2 * node + 1 + go_right
    return leaves[node - (2**depth - 1)]


class GradientBoosting:
    """Boosted trees on the logistic loss.

    Default hyper-parameters were chosen by cross-validation: with learning
    rate 0.1 and depth 5, validation F1 peaks around 150 trees and then
    slowly decreases (overfitting).
    """

    def __init__(
        self,
        n_trees=150,
        depth=5,
        learning_rate=0.1,
        lam=1.0,
        min_hessian=20.0,
        row_sample=0.5,
        feature_sample=0.5,
        seed=0,
    ):
        self.n_trees = n_trees
        self.depth = depth
        self.learning_rate = learning_rate
        self.lam = lam
        self.min_hessian = min_hessian
        self.row_sample = row_sample
        self.feature_sample = feature_sample
        self.seed = seed

    def fit(self, x, y):
        """Fit on raw features x (N, F) with NaNs and labels y in {0, 1}."""
        rng = np.random.default_rng(self.seed)
        self.edges = fit_bins(x)
        xb = apply_bins(x, self.edges)
        # start from the log-odds of the positive rate
        self.base = np.log(y.mean() / (1 - y.mean()))
        score = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.n_trees):
            p = sigmoid(score)
            g = p - y  # gradient of the logistic loss w.r.t. the score
            h = p * (1 - p)  # its second derivative
            rows = rng.random(len(y)) < self.row_sample
            features = rng.random(x.shape[1]) < self.feature_sample
            splits, leaves = build_tree(
                xb[rows], g[rows], h[rows], self.depth, self.lam,
                self.min_hessian, features,
            )
            # Shrink each tree by the learning rate so later trees can
            # correct it; the score is updated on ALL rows, not just the
            # sampled ones.
            leaves = self.learning_rate * leaves
            score += predict_tree(xb, splits, leaves, self.depth)
            self.trees.append((splits, leaves))
        return self

    def predict_proba(self, x):
        """Return P(y = 1) for every row of the raw features x."""
        xb = apply_bins(x, self.edges)
        score = np.full(len(x), self.base)
        for splits, leaves in self.trees:
            score += predict_tree(xb, splits, leaves, self.depth)
        return sigmoid(score)
