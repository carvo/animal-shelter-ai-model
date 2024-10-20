import logging
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from animal_shelter.feature.default_features import DefaultFeatures
from animal_shelter.feature.enhancer import add_features
from animal_shelter.helper.data_loader import convert_camel_case, standardize
from animal_shelter.model.domain import AnimalPrediction, ListAnimalPrediction

LOG = logging.getLogger(__name__)


def predict_file(data: bytes, model_path: Path) -> pd.DataFrame:
    raw_data = standardize(pd.read_csv(BytesIO(data)))
    return predict(raw_data, model_path)


def predict_json(data: AnimalPrediction, model_path: Path) -> pd.DataFrame:
    dumped_model = [data.model_dump()]
    raw_data = pd.DataFrame.from_dict(dumped_model)
    return predict(raw_data, model_path)


def predict_json_list(data: ListAnimalPrediction, model_path: Path) -> pd.DataFrame:
    dumped_models = list(map((lambda x: x.model_dump()), data.predictions))
    raw_data = pd.DataFrame.from_records(dumped_models)
    return predict(raw_data, model_path)


def predict(raw_data: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Generate predictions on the provided data.
    :data: path to the data
    :model_path: which model to use
    """
    LOG.debug("Using model %s", model_path)

    with_features = add_features(raw_data)
    x = with_features[DefaultFeatures.CATEGORY_FEATURES + DefaultFeatures.NUM_FEATURES]

    model = _load_model(model_path)
    y_pred = model.predict_proba(x)

    # Combine predictions with class names and animal name.
    classes = model.classes_.tolist()
    proba_df = pd.DataFrame(y_pred, columns=classes).rename(str.lower, axis=1)

    return raw_data[["id"]].join(raw_data[["name"]]).join(proba_df)


def _load_model(model_path: Path) -> Pipeline:
    """Load the model from the given path
    :param model_path: path to the model
    :return: model pipeline
    """
    # This function could point to an experiment tracking system instead of to a local serialized model
    return joblib.load(model_path)
