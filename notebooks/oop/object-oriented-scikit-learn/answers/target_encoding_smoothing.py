class TargetEncoderSmoothing(BaseEstimator, TransformerMixin):
    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        self.n = len(y)
        self.w = y.mean()

        # Create ordinal encoder to convert from strings and whatnot to numeric values
        self.ordinal_ = OrdinalEncoder()
        X_ = self.ordinal_.fit_transform(X)

        # Create mapping from column-unique value to probability
        self.mapping_ = []
        for col in range(X_.shape[1]):
            column_mapping = []
            column = X_[:, col]
            for value in np.unique(column):
                targets = y[column == value]
                estimated_mean = targets.sum() / len(targets)
                mu = (self.n * estimated_mean + self.m * self.w) / (self.n + self.m)
                column_mapping.append(mu)
            self.mapping_.append(column_mapping)
        return self
        pass

    def transform(self, X):
        X_ = self.ordinal_.transform(X)
        for col in range(X_.shape[1]):
            column = X_[:, col]
            for value in np.unique(column):
                column[column == value] = self.mapping_[col][int(value)]
        return X_


# Convert to NumPy to prove that it works
array = df.to_numpy()
X = array[:, :-1]
y = array[:, -1]

# Encode
encoder = TargetEncoderSmoothing(0.2)
encoder.fit(X, y)
encoder.transform(X)
