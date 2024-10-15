from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression


class VarianceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        X_ = X.copy()
        _, p_values = f_regression(X_, y)
        self.selected_mask_ = p_values <= self.threshold
        return self

    def transform(self, X, y=None):
        X = X.copy().to_numpy()
        return X[:, self.selected_mask_]
