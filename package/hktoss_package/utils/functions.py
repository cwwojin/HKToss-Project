from pandas import Series, DataFrame
from typing import Union


def is_bool_col(column: Union[Series, DataFrame]):
    return set(column.unique()) == {0, 1}
