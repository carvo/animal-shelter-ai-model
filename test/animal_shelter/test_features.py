from pandas.testing import assert_series_equal
import pandas as pd
from animal_shelter import features

def test_check_has_name():
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features.check_has_name(s)

    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)
