# -*- coding=utf-8 -*-
# @Project  : 20240720
# @FilePath : vnpy/vnpy/utils
# @File     : polars_utils.py
# @Time     : 2025/8/10 16:32
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Description:

import polars as pl
from polars._typing import ColumnNameOrSelector
import traceback


# This function was written by Gemini
def transpose_dataframe(
    df: pl.DataFrame,
    index: str | list[str] = None,
    on: str | list[str] | ColumnNameOrSelector|list[ColumnNameOrSelector] = None,
    variable_name: str = "variable",
    value_name: str = "value",
) -> pl.DataFrame:
    """
    Transposes a Polars DataFrame, keeping the specified index column as is.
    The values of the index column will become the new header.
    
    Args:
        df (pl.DataFrame): The DataFrame to transpose.
        index (str | list[str]): Column(s) to use as identifier variables.
        on (str | list[str], optional): Columns to unpivot. Defaults to all columns that are not in `index`.
        variable_name (str, optional): Name to give to the 'variable' column. Defaults to "variable".
        value_name (str, optional): Name to give to the 'value' column. Defaults to "value".
        
    Returns:
        pl.DataFrame: The transposed DataFrame.
    """
    try:
        # The following logic was adjusted by Gemini
        df_unpivoted = df.unpivot(index=index, on=on, variable_name=variable_name, value_name=value_name)
        df_pivoted = df_unpivoted.pivot(index=variable_name, columns=index, values=value_name)
        return df_pivoted

    except Exception as e:
        traceback.print_tb(e.__traceback__)
