from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor()

pipeline = Pipeline(
    steps=[
        ("VT", VarianceThreshold(0.05)),
        ("scaler", MinMaxScaler()),
        ("model", model),
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
