from sklearn.preprocessing import OrdinalEncoder


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
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
                probabilities = targets.sum() / len(targets)
                column_mapping.append(probabilities)
            self.mapping_.append(column_mapping)
        return self

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
encoder = TargetEncoder()
encoder.fit(X, y)
encoder.transform(X)
