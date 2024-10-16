import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

from animal_shelter.data_loader import load_data
from animal_shelter.features import add_features

project_root_path = Path(__file__).parent.parent.parent.parent
print(project_root_path)

csv_train_file = project_root_path / "data/train.csv"
raw_data = load_data(csv_train_file.__str__())

with_features = add_features(raw_data)
cat_features = [
    "animal_type",
    "is_dog",
    "has_name",
    "sex",
    "hair_type",
]
num_features = ["days_upon_outcome"]

num_transformer = Pipeline(
    steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
)
cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])
transformer = ColumnTransformer(
    [
        ("numeric", num_transformer, num_features),
        ("categorical", cat_transformer, cat_features),
    ]
)

clf_model = Pipeline(
    [("transformer", transformer), ("model", RandomForestClassifier())]
)

X = with_features[cat_features + num_features]
y = with_features["outcome_type"]

clf_model.fit(X, y)
