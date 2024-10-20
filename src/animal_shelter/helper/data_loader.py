import logging
import pandas as pd
import re

from pathlib import Path

LOG = logging.getLogger(__name__)


def load_data(file_path: Path):
    """Load the data and convert the column names.

    Parameters
    ----------
    file_path : Path
        Path to data relative to the PROJECT path
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with data
    """
    LOG.debug(f"Loading data from {file_path}")
    df = standardize(pd.read_csv(file_path, parse_dates=["DateTime"]))

    return df


def convert_camel_case(name):
    """Convert camelCaseString to snake_case_string."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .assign(date=lambda d: pd.to_datetime(d['DateTime']).dt.normalize())
        .rename(columns=lambda x: x.replace("upon", "Upon"))
        .rename(columns=convert_camel_case)
        .fillna("Unknown")
    )
