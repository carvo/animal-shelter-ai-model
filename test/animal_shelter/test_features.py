import pytest
from pandas.testing import assert_series_equal
import pandas as pd
from animal_shelter import features
import logging

LOG = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def animal_types_series():
    return pd.Series(["dOg", "caT"])


def test_check_is_dog_raises_runtime(animal_types_series):
    with pytest.raises(RuntimeError):
        features.check_is_dog(animal_types_series.add("ET"))


def test_check_is_dog(animal_types_series):
    result = features.check_is_dog(animal_types_series)

    expected = pd.Series([True, False])
    assert_series_equal(result, expected)


def test_check_has_name():
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features.check_has_name(s)

    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)


def test_get_sex():
    s = pd.Series(["Female", "Female", "Male"])
    result = features.get_sex(s)

    expected = pd.Series(["female", "female", "male"])
    assert_series_equal(result, expected)


def test_get_neutered():
    s = pd.Series(["neutered", "spayed", "intact", "trouxa", "titio"])
    result = features.get_neutered(s)

    expected = pd.Series(["fixed", "fixed", "intact", "unknown", "unknown"])
    assert_series_equal(result, expected)
