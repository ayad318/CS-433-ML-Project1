"""Feature preprocessing for the BRFSS heart-attack (MICHD) dataset.

The raw data comes from the CDC Behavioral Risk Factor Surveillance System.
Most columns are coded survey answers where 7/9, 77/99, 777/999, ... mean
"don't know" or "refused", and 88/888 mean "none". We therefore:

  * drop administrative columns (dates, sequence numbers, survey weights) and
    raw answers that BRFSS already provides in a cleaned, computed form;
  * one-hot encode every low-cardinality (categorical) column, keeping the
    missing value and the "don't know"/"refused" codes as their own levels;
  * for numeric columns, map refusal codes to NaN and "none" codes to 0,
    log-transform heavy-tailed columns, impute the median, add a missing
    indicator and standardize.

All statistics (categories, medians, means, stds) are fitted on the training
set only and then applied unchanged to the test set.
"""

import numpy as np

# Columns that carry no information about the respondent's health
# (interview metadata, identifiers, survey weights).
DROP_COLUMNS = {
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALCOR",
    "_LLCPWT", "QSTVER", "QSTLANG",
    # Raw answers with a mixed unit encoding (e.g. 1xx = per week,
    # 2xx = per month). BRFSS ships cleaned versions of each of them
    # (WTKG3, HTM4, DROCDY3_, FRUTDA1_, ..., PAFREQ1_, PADUR1_, STRFREQ_).
    "WEIGHT2", "HEIGHT3", "ALCDAY5", "FRUITJU1", "FRUIT1", "FVBEANS",
    "FVGREEN", "FVORANG", "VEGETAB1", "EXEROFT1", "EXEROFT2", "EXERHMM1",
    "EXERHMM2", "STRENGTH", "FLSHTMY2", "HIVTSTD3",
}

# Columns with at most this many distinct values are treated as categorical.
MAX_CATEGORIES = 12

# Categories seen fewer times than this in training are merged into "other".
MIN_CATEGORY_COUNT = 50

# Refusal / don't-know codes for numeric columns, keyed by the column maximum.
# The maximum tells us the width of the answer field: e.g. PHYSHLTH (days of
# poor health, 1-30) has max 99, so 77 = "don't know" and 99 = "refused";
# computed fields like DROCDY3_ use 900 / 99900 as their missing marker.
MISSING_CODES = {
    99: (77, 99),
    98: (77, 99),
    999: (777, 999),
    900: (900,),
    9999: (7777, 9999),
    99900: (99900,),
}
# "None" codes, e.g. PHYSHLTH = 88 means zero days of poor health. They are a
# real value (0), not a missing answer.
NONE_CODES = {99: 88, 999: 888}


def load_column_names(path):
    """Return the feature names of an x_*.csv file (without the Id column)."""
    with open(path) as f:
        return f.readline().strip().split(",")[1:]


class Preprocessor:
    """Fit-on-train / transform-anywhere feature pipeline.

    Usage:
        pp = Preprocessor().fit(x_train, names)
        features_train = pp.transform(x_train)
        features_test = pp.transform(x_test)  # uses the training statistics
    """

    def fit(self, x, names):
        """Learn the column types and all statistics from the training data.

        Args:
            x: numpy array of shape (N, 321), raw features (NaN = not asked).
            names: list of the 321 column names, in the same order as x.

        Returns:
            self, so that fit and transform can be chained.
        """
        self.keep = [i for i, n in enumerate(names) if n not in DROP_COLUMNS]
        self.categorical = []  # (column index, list of category values)
        self.numeric = []  # (column index, missing codes, none code, use log)
        for i in self.keep:
            col = x[:, i]
            values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
            if len(values) == 0:  # column is entirely missing
                continue
            # _STATE has 53 values but is a label, not a quantity.
            if len(values) <= MAX_CATEGORIES or names[i] == "_STATE":
                # Rare levels would get unreliable weights, so they are
                # merged into a single "other" indicator in transform.
                frequent = values[counts >= MIN_CATEGORY_COUNT]
                self.categorical.append((i, frequent))
            else:
                vmax = values.max()
                missing = MISSING_CODES.get(int(vmax), ())
                none = NONE_CODES.get(int(vmax))
                clean = self._clean(col, missing, none)
                ok = clean[~np.isnan(clean)]
                # Heavy-tailed, non-negative columns (minutes of exercise,
                # drinks per week, ...) are log-compressed so that a few
                # extreme answers do not dominate the linear model.
                use_log = ok.min() >= 0 and np.median(ok) * 5 < ok.max()
                self.numeric.append((i, missing, none, use_log))

        # Imputation and standardization statistics, from training data only.
        num = self._numeric_raw(x)
        self.median = np.nanmedian(num, axis=0)
        filled = np.where(np.isnan(num), self.median, num)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0) + 1e-8  # avoids division by zero
        return self

    @staticmethod
    def _clean(col, missing, none):
        """Map refusal codes to NaN and the "none" code to 0 in one column."""
        col = col.copy()
        for code in missing:
            col[col == code] = np.nan
        if none is not None:
            col[col == none] = 0
        return col

    def _numeric_raw(self, x):
        """Return the cleaned (and log-transformed) numeric columns of x."""
        out = np.empty((x.shape[0], len(self.numeric)))
        for j, (i, missing, none, use_log) in enumerate(self.numeric):
            col = self._clean(x[:, i], missing, none)
            out[:, j] = np.log1p(col) if use_log else col
        return out

    def transform(self, x):
        """Turn raw features into the model's feature matrix.

        Args:
            x: numpy array of shape (N, 321), raw features.

        Returns:
            numpy array of shape (N, D) with D around 1300: standardized
            numeric features, their missing indicators and the one-hot
            encoded categorical features. No bias column is added.
        """
        num = self._numeric_raw(x)
        is_missing = np.isnan(num)
        num = (np.where(is_missing, self.median, num) - self.mean) / self.std

        # The missing indicators let the model learn that "not answered"
        # differs from "answered the median value".
        blocks = [num, is_missing.astype(float)]
        for i, cats in self.categorical:
            col = x[:, i]
            onehot = (col[:, None] == cats[None, :]).astype(float)
            # NaN = question not asked (survey skip logic), kept as a level.
            nan = np.isnan(col)[:, None].astype(float)
            # "other": a value that is present but rare / unseen in training
            other = 1.0 - onehot.sum(axis=1, keepdims=True) - nan
            blocks += [onehot, nan, other]
        features = np.hstack(blocks)
        # The first call (on the training data, right after fit) decides
        # which features are constant; later calls reuse that mask so train
        # and test always have the same columns.
        if not hasattr(self, "nonconstant"):
            self.nonconstant = features.std(axis=0) > 0
        return features[:, self.nonconstant]

    def fit_transform(self, x, names):
        """Fit on x, then transform x."""
        return self.fit(x, names).transform(x)
