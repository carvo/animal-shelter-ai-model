from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        sums = np.sum(X, axis=0)
        self.columns_keep_ = np.where(~(sums == 0) & ~(sums == 5))[0]
        return self

    def transform(self, X):
        return X[:, self.columns_keep_]


transformer = CustomTransformer()
transformer.fit(X)
