import pandas as pd
import numpy as np


def handle_df_nulls(data, how, column_names):
    '''
    Description: Processes the null values in the dataset
    Input:
    data: dataframe with missing values
    how: processing method, currently supports
            - 'special': corresponds to 'not applicable' scenario, designates null values as their own special category
            - 'drop-column' : removes the column with nulls from the dataset
            - 'drop-rows' : removes all the rows with the nulls values from the dataset
    column-names: list of column names, for which the particular techniques needs to be applied

    Output:
    dataframe with processed nulls
    '''
    if how == 'special':
        vals = {}
        for col in column_names:
            vals[col] = decide_special_category(data[col].values)
        data.fillna(value=vals, inplace=True)
    
    if how == 'drop-column':
        data.drop(columns=column_names, inplace=True)

    if how == 'drop-rows':
        data.dropna(subset=column_names, inplace=True)
        
    return data


def get_sample_rows(data, target_col, fraction):
    """
    Description: create a list of random indexes for rows, which will be used to place nulls in the dataset
    """
    n_values_to_discard = int(len(data) * min(fraction, 1.0))
    perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
    perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

    depends_on_col = np.random.choice(list(set(data.columns) - {target_col}))
    # Pick a random percentile of values in other column
    rows = data[depends_on_col].sort_values().iloc[perc_idx].index
    return rows


def nulls_simulator(data, target_col, condition_col, special_values, fraction, nan_value=np.nan):
    """
    Description: simulate nulls for the target column in the dataset based on the condition column and its special values.

    Input:
    :param data: a dataset, in which nulls values should be simulated
    :param target_col: a column in the dataset, in which nulls should be placed
    :param condition_col: a column in the dataset based on which null location should be identified
    :param special_values: special values for the condition column; special_values and condition_col state the condition,
        where nulls should be placed
    :param fraction: float in range [0.0, 1.0], fraction of nulls, which should be placed based on the condition
    :param nan_value: a value, which should be used as null to be placed in the dataset

    Output: a dataset with null values based on the condition and fraction
    """
    if target_col not in data.columns:
        return ValueError(f'Column {target_col} does not exist in the dataset')
    if condition_col not in data.columns:
        return ValueError(f'Column {condition_col} does not exist in the dataset')
    if not (0 <= fraction <= 1):
        return ValueError(f'Fraction {fraction} is not in range [0, 1]')

    corrupted_data = data[data[condition_col].isin(special_values)].copy(deep=True)
    rows = get_sample_rows(corrupted_data, target_col, fraction)
    corrupted_data.loc[rows, [target_col]] = nan_value
    corrupted_data = pd.concat([corrupted_data, data[~data[condition_col].isin(special_values)]], axis=0)
    return corrupted_data


def decide_special_category(data):
    '''
    Description: Decides which value to designate as a special value, based on the values already in the data
    '''
    if 0 not in data:
        return 0
    else:
        return max(data) + 1