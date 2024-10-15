from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sums = np.sum(X, axis=0)
        idx = np.where(~(sums == 0) & ~(sums == 5))[0]
        X_columns_removed = X[:, idx]
        return X_columns_removed


transformer = CustomTransformer()
