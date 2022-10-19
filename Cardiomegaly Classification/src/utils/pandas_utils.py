'''  This file contains more general functions that are used within nodes
'''

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def LabelRatio_y(y):
    '''
    Calculate percentage of +ve cases for a list of numbers e.g [0,1,1,1,0]
    '''
    ratio = sum(y) / len(
        y
    )  # Ratio is just the sum (number of positive cases) over the length (number of all cases)
    return ratio


def explode(
    df: pd.DataFrame, lst_cols: List[str], fill_value: str = ''
) -> pd.DataFrame:
    '''Explode function for exploding dataframes based on two columns
    (The default pandas function only allows for exploding based on one column)

    :param df: Dataframe to explode
    :type df: pd.DataFrame
    :param lst_cols: Column list
    :type lst_cols: List[str]
    :param fill_value: Fill value, defaults to ''
    :type fill_value: str, optional
    :return: Adjusted DataFrame
    :rtype: pd.DataFrame
    '''

    # We need to use this function instead of the default panda explode() function because if you choose to do
    # hourly averages (so each row is an hourly average of ICU data) you need to explode based on two columns (Times,
    # and Charttime).
    # I didn't come up with this function, it was suggested here:
    # https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows

    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        df_return = (
            pd.DataFrame(
                {
                    col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                    for col in idx_cols
                }
            )
            .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
            .loc[:, df.columns]
        )
    else:
        # at least one list in cells is empty
        df_return = (
            pd.DataFrame(
                {
                    col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                    for col in idx_cols
                }
            )
            .assign(**{col: np.concatenate(df[col].values) for col in lst_cols})
            .append(df.loc[lens == 0, idx_cols])
            .fillna(fill_value)
            .loc[:, df.columns]
        )

    return df_return


def create_pivot(
    df: pd.DataFrame,
    labels: Dict[int, str],
    labels_column: str,
    unique_matcher: List[str],
    aggfunc: str,
    values: str,
) -> pd.DataFrame:
    '''Create pivot table using pandas.pivot_table after filtering for labels and renaming labels at the end

    :param df: DataFrame
    :type df: pd.DataFrame
    :param labels: Labels
    :type labels: Dict[int, str]
    :param labels_column: Column with labels to filter for and to use for columns in pd.pivot_table
    :type labels_column: str
    :param unique_matcher: index in pd.pivot_table
    :type unique_matcher: List[str]
    :param aggfunc: aggfunc in pd.pivot_table
    :type aggfunc: str
    :param values: values in pd.pivot_table
    :type values: str
    :return: Pivoted dataframe
    :rtype: pd.DataFrame
    '''

    df_filtered = df.loc[df[labels_column].isin(labels.keys())]
    df_pivoted = pd.pivot_table(
        df_filtered,
        values=values,
        index=unique_matcher,
        columns=[labels_column],
        aggfunc=aggfunc,
    )
    df_pivoted = df_pivoted.reset_index()
    df_pivoted = df_pivoted.rename(columns=labels)

    return df_pivoted


def filter_df_isin(
    df: pd.DataFrame,
    filter_col: str,
    filter_list: List[any],
) -> pd.DataFrame:
    '''Filter pandas DataFrame via isin()

    :param df: DataFrame to filter
    :type df: pd.DataFrame
    :param filter_col: Column to filter by
    :type filter_col: str
    :param filter_list: List to filter for
    :type filter_list: List[any]
    :return: Filtered DataFrame
    :rtype: pd.DataFrame
    '''

    return df[df[filter_col].isin(filter_list)]


def list_of_dicts_to_dict_of_list(
    dict_list: List[Dict[str, Any]], search_key: str
) -> Dict[str, List[Any]]:
    '''Convert the values of the key search_key in a list of dictionaries to a dictionary containing for all keys under the value of search_key all values as a list

    :param dict_list: Original list of dictionaries with a item with key search_key and a dictionary as value
    :type dict_list: List[Dict[str, Any]]
    :param search_key: Key to look for
    :type search_key: str
    :return: Dictionary containing lists of all values in dict_list sorted by the keys of the dictionary under the dictionary of search_key
    :rtype: Dict[str, List[Any]]
    '''

    list_dict = {k: [] for k in dict_list[0][search_key].keys()}
    list_dict['fold'] = []
    for index in range(len(dict_list)):
        for key, value in dict_list[index][search_key].items():
            list_dict[key].append(value)
        list_dict['fold'].append(index)

    return list_dict
