"""Data loading and preprocessing for the BRFSS heart-disease dataset.

The pipeline has two stages:

1. `clean_raw` applies fixed, column-wise rules that do not learn anything
   from the data (drop administrative columns, turn BRFSS "don't know /
   refused" codes into NaN, map "none" codes to 0). It can be applied to
   train and test independently.

2. `Preprocessor` learns statistics on the TRAINING data only (which columns
   to keep, feature types, medians, means/stds, categories) and applies them
   to any other split. In cross-validation it is re-fitted on every training
   fold so that no information leaks from the validation fold.

Only NumPy and the standard library are used.
"""

import os

import numpy as np

from helpers import load_csv_data


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_column_names(data_path):
    """Return the 321 feature names of x_train.csv (without the Id column)."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def load_data(data_path="dataset"):
    """Load the dataset, caching it as .npy files after the first CSV parse.

    Parsing the CSVs takes ~2 minutes; loading the .npy cache takes seconds.
    The cache lives next to the CSVs (inside dataset/, which is git-ignored).

    Returns:
        x_train, x_test, y_train (in {-1, 1}), train_ids, test_ids, columns
    """
    names = ["x_train", "x_test", "y_train", "train_ids", "test_ids"]
    paths = [os.path.join(data_path, n + ".npy") for n in names]
    if all(os.path.exists(p) for p in paths):
        arrays = [np.load(p) for p in paths]
    else:
        arrays = load_csv_data(data_path)
        for p, a in zip(paths, arrays):
            np.save(p, a)
    return (*arrays, load_column_names(data_path))


# ---------------------------------------------------------------------------
# Stage 1: fixed cleaning rules
# ---------------------------------------------------------------------------

# Survey bookkeeping: interview dates, sampling design and weights, phone
# screening questions. They describe how the interview was done, not the
# respondent's health.
ADMIN_COLUMNS = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE",
    "LANDLINE", "HHADULT", "NUMHHOL2", "NUMPHON2", "CPDEMO1", "QSTVER",
    "QSTLANG", "MSCODE", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
    "_CHISPNC", "_DUALUSE", "_DUALCOR", "_LLCPWT",
]

# Raw answers that pack a unit into the value (e.g. 1xx = per day,
# 2xx = per week) or mix units (pounds/kg, ft-in/cm). BRFSS provides cleaned,
# computed versions of all of them, which we keep instead:
#   fruit/veg -> FTJUDA1_, FRUTDA1_, ...   ALCDAY5 -> DROCDY3_, _DRNKWEK
#   EXEROFT/EXERHMM -> PAFREQ*_, PADUR*_    STRENGTH -> STRFREQ_
#   WEIGHT2/HEIGHT3 -> WTKG3, HTM4, _BMI5   EXRACT* (activity type) dropped
ENCODED_RAW_COLUMNS = [
    "ALCDAY5", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG",
    "VEGETAB1", "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2", "STRENGTH",
    "EXRACT11", "EXRACT21", "WEIGHT2", "HEIGHT3",
]

# Missing-value codes the generic rule below does not catch.
EXTRA_MISSING_CODES = {"DROCDY3_": [900], "_DRNKWEK": [99900]}

# Columns with many levels that are nominal, not ordered (one-hot encode).
FORCE_CATEGORICAL = ["_STATE"]

_ALL_NINES = {9, 99, 999, 9999}


def clean_raw(x, columns):
    """Apply the fixed cleaning rules. Returns (x_clean, kept_columns).

    BRFSS uses the largest "all nines" value of a question for "refused" and
    the same width of sevens for "don't know" (9/7, 99/77, 999/777, ...), and
    eights for "none" (88, 888). The generic rule, applied per column, is:
      - if the column maximum M is 9, 99, 999 or 9999, M becomes NaN;
      - the sevens code D = 7M/9 (7, 77, 777, ...) becomes NaN if D - 1 does
        not occur, i.e. it is a code sitting apart from the real range (so
        EMPLOY1 = 7 "retired", which follows 1..6, is kept);
      - for M >= 99, the eights code 8M/9 (88, 888, ...) becomes 0 ("none")
        under the same isolation condition, e.g. PHYSHLTH = 88 means zero
        bad days.

    The rules are computed from the column given, so call it on the
    concatenation of train and test if you want identical decisions on both
    (run.py does). The decisions only depend on which codes exist.

    The result is float32 to halve memory use: after dropping the IDs and
    weights, every remaining value is a small integer code or a measurement
    with at most 2 decimals, which float32 represents accurately.
    """
    drop = set(ADMIN_COLUMNS) | set(ENCODED_RAW_COLUMNS)
    keep = [j for j, c in enumerate(columns) if c not in drop]
    x = x[:, keep].astype(np.float32)
    columns = [columns[j] for j in keep]

    for j, c in enumerate(columns):
        v = x[:, j]
        present = set(np.unique(v[~np.isnan(v)]).tolist())
        if not present:
            continue
        m = max(present)
        if m in _ALL_NINES:
            dont_know, none = m * 7 // 9, m * 8 // 9  # 99 -> 77, 88
            v[v == m] = np.nan
            if dont_know in present and (dont_know - 1) not in present:
                v[v == dont_know] = np.nan
            if m >= 99 and none in present and (none - 1) not in present:
                v[v == none] = 0
        for code in EXTRA_MISSING_CODES.get(c, []):
            v[v == code] = np.nan
    return x, columns


# ---------------------------------------------------------------------------
# Stage 2: learned preprocessing
# ---------------------------------------------------------------------------


class Preprocessor:
    """Turn cleaned raw columns into a numeric design matrix.

    Steps, all learned on the data passed to `fit`:
      1. Drop columns with more than `max_nan_frac` missing values and
         columns that are constant.
      2. Columns with at most `max_categories` distinct values (or listed in
         FORCE_CATEGORICAL) are categorical: one-hot encoded, dropping the
         most frequent level as reference to avoid collinearity with the
         bias column.
      3. Other columns are continuous: optionally log1p-transformed when
         non-negative and strongly right-skewed, imputed with the median and
         standardized to zero mean and unit variance.
      4. If `missing_indicators`, add one 0/1 column per distinct
         missingness pattern ("was this value missing?"). Questions from the
         same survey module share a pattern, so duplicates are removed.
      5. Prepend a column of ones (bias).
    """

    def __init__(self, max_nan_frac=0.8, max_categories=10,
                 log_skewed=True, missing_indicators=True):
        self.max_nan_frac = max_nan_frac
        self.max_categories = max_categories
        self.log_skewed = log_skewed
        self.missing_indicators = missing_indicators

    def fit(self, x, columns):
        nan = np.isnan(x)
        nan_frac = nan.mean(axis=0)
        self.columns_in = list(columns)
        self.categorical = []  # (col index, levels to one-hot)
        self.continuous = []   # (col index, use_log, median, mean, std)
        self.feature_names = ["bias"]

        for j, c in enumerate(columns):
            if nan_frac[j] > self.max_nan_frac:
                continue
            v = x[~nan[:, j], j]
            levels, counts = np.unique(v, return_counts=True)
            if len(levels) < 2:
                continue
            if len(levels) <= self.max_categories or c in FORCE_CATEGORICAL:
                kept = np.delete(levels, np.argmax(counts))  # drop reference
                self.categorical.append((j, kept))
                self.feature_names += [f"{c}={lv:g}" for lv in kept]
            else:
                use_log = self.log_skewed and v.min() >= 0 and _skew(v) > 2
                if use_log:
                    v = np.log1p(v)
                std = v.std()
                if std == 0:
                    continue
                self.continuous.append((j, use_log, np.median(v), v.mean(), std))
                self.feature_names.append(f"log1p({c})" if use_log else c)

        # Missingness indicators: one per distinct pattern among used columns.
        self.indicator_cols = []
        if self.missing_indicators:
            used = sorted({j for j, _ in self.categorical} |
                          {j for j, *_ in self.continuous})
            cand = [j for j in used if 0 < nan_frac[j] < 1]
            if cand:
                packed = np.packbits(nan[:, cand], axis=0).T  # one row per col
                _, first = np.unique(packed, axis=0, return_index=True)
                self.indicator_cols = [cand[i] for i in sorted(first)]
            self.feature_names += [f"missing({columns[j]})"
                                   for j in self.indicator_cols]
        return self

    def transform(self, x, dtype=np.float32):
        n = x.shape[0]
        out = np.empty((n, len(self.feature_names)), dtype=dtype)
        out[:, 0] = 1.0
        k = 1
        for j, levels in self.categorical:
            out[:, k:k + len(levels)] = x[:, j:j + 1] == levels[None, :]
            k += len(levels)
        for j, use_log, median, mean, std in self.continuous:
            v = x[:, j]
            if use_log:
                v = np.log1p(np.clip(v, 0, None))
            v = np.where(np.isnan(v), median, v)
            out[:, k] = (v - mean) / std
            k += 1
        for j in self.indicator_cols:
            out[:, k] = np.isnan(x[:, j])
            k += 1
        return out

    def fit_transform(self, x, columns, dtype=np.float32):
        return self.fit(x, columns).transform(x, dtype)


def _skew(v):
    """Sample skewness (third standardized moment)."""
    s = v.std()
    return 0.0 if s == 0 else float(np.mean(((v - v.mean()) / s) ** 3))
