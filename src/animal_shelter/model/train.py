from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from animal_shelter.data_loader import load_data
from animal_shelter.features import add_features
from animal_shelter.model.default_features import DefaultFeatures

project_root_path = Path(__file__).parent.parent.parent.parent

def full_train(input_path):
    raw_data = _load_raw_data(input_path)
    data_with_features = add_features(raw_data)

    num_transformer = Pipeline([
        ("imputer", SimpleImputer()), ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("onehot", OneHotEncoder(drop="first"))
    ])
    col_transformer = ColumnTransformer([
        ("numeric", num_transformer, DefaultFeatures.cat_features),
        ("categorical", cat_transformer, DefaultFeatures.num_features),
    ])

    x = data_with_features[DefaultFeatures.cat_features + DefaultFeatures.num_features]
    y = data_with_features["outcome_type"]

    return Pipeline([
        ("col_transformer", col_transformer), ("model", RandomForestClassifier())
    ]).fit(x, y)

def _load_raw_data(input_path):
    csv_train_file = project_root_path / input_path
    return load_data(csv_train_file.__str__())
