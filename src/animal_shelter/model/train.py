import logging
import string
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

from animal_shelter.helper.data_loader import load_data
from animal_shelter.feature.enhancer import add_features
from animal_shelter.feature.default_features import DefaultFeatures

LOG = logging.getLogger(__name__)


def train(data_path: string, output_path: Path):
    raw_data = load_data(data_path)
    data_with_features = add_features(raw_data)

    x = data_with_features[DefaultFeatures.CATEGORY_FEATURES + DefaultFeatures.NUM_FEATURES]
    y = data_with_features["outcome_type"]

    model = _fit_model(_build_pipeline(), x, y)
    _save_model(model, output_path)

    return model

def _build_pipeline(encoder_drop="first"):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer()), ("scaler", StandardScaler())
    ])
    category_transformer = Pipeline([
        ("onehot", OneHotEncoder(drop=encoder_drop))
    ])
    col_transformer = ColumnTransformer([
        ("numeric", num_transformer, DefaultFeatures.NUM_FEATURES),
        ("categorical", category_transformer, DefaultFeatures.CATEGORY_FEATURES),
    ])

    return Pipeline([
        ("col_transformer", col_transformer), ("model", RandomForestClassifier())
    ])

def _fit_model(model: Pipeline, x: pd.DataFrame, y: pd.Series):
    """Train the model
    :param model: model pipeline
    :param x: feature
    :param y: target variable
    :return: trained model pipeline
    """
    return model.fit(x, y)

def _save_model(model: Pipeline, path: Path) -> None:
    """Save the model.
    :param model: model object
    :param path: path to the model
    """
    LOG.info("Saving model at %s", path)
    joblib.dump(model, path)
