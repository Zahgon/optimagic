
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from pybaum import get_registry as get_pybaum_registry


def get_registry(extended=False, data_col="value"):
    """Return pytree registry.

    Special Rules
    -------------
    If extended is True the registry contains pd.DataFrame. In optimagic a data frame
    can represent a 1d object with extra information, instead of a 2d object. This is
    only allowed for params data frames, in which case they contain a 'value' column.
    The extra information of such an object can be accessed using the data_col argument.
    By default the 'value' column is extracted. If data_col is not 'value' but the data
    frame contains a 'value' column, a list of np.nan is returned.

    Args:
        extended (bool): If True appends types 'numpy.ndarray', 'pandas.Series' and
            'pandas.DataFrame' to the registry.
        data_col (str): This column is used as the data source in a data frame when
            flattening and unflattening a pytree. Defaults to 'value'; see special rules
            above for behavior with non-default values.

    Returns:
        dict: The pytree registry.

    """
    types = (
        ["numpy.ndarray", "pandas.Series", "jax.numpy.ndarray"] if extended else None
    )
    registry = get_pybaum_registry(types=types)
    if extended:
        registry[pd.DataFrame] = {
            "flatten": partial(_flatten_df, data_col=data_col),
            "unflatten": partial(_unflatten_df, data_col=data_col),
            "names": _get_df_names,
        }
    return registry


def _flatten_df(df, data_col):
    pass


def _unflatten_df(aux_data, leaves, data_col):
    pass


def _get_df_names(df):
    pass


def _index_element_to_string(element):
    pass
