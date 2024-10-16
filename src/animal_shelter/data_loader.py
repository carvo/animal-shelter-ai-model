from pathlib import Path
import pandas as pd
import re

project_root_path = Path(__file__).parent.parent

def load_data(path):
    """Load the data and convert the column names.

    Parameters
    ----------
    path : str
        Path to data relative to the PROJECT path
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with data
    """
    file_path = project_root_path / path
    df = (
        pd.read_csv(file_path.__str__(), parse_dates=["DateTime"])
        .rename(columns=lambda x: x.replace("upon", "Upon"))
        .rename(columns=convert_camel_case)
        .fillna("Unknown")
    )
    return df


def convert_camel_case(name):
    """Convert camelCaseString to snake_case_string."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
