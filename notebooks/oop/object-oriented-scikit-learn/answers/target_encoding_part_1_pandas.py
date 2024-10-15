# Your code.
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoderPandas(BaseEstimator, TransformerMixin):
    """This target encoder assumes the input to be a Pandas DataFrame"""

    def __init__(self):
        pass

    def fit(self, X, y):
        df = pd.concat([X, y], axis=1)
        self.mappings_ = {}
        label = y.name
        for col in X:
            means = df.groupby(col)[label].mean()
            self.mappings_[col] = means
        return self

    def transform(self, X):
        X = X.copy()
        for col in X:
            X[col] = X[col].map(self.mappings_[col])
        return X


# Create X and y as Pandas DataFrames.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode
encoder = TargetEncoderPandas()
encoder.fit(X, y)
encoder.transform(X)
