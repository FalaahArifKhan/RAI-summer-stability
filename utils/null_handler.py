import pandas as pd
import numpy as np

from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def handle_df_nulls(input_data, how, column_names, condition_column=None):
    """
    Description: Processes the null values in the dataset
    Input:
    data: dataframe with missing values
    how: processing method, currently supports
            - 'special': corresponds to 'not applicable' scenario, designates null values as their own special category
            - 'drop-column' : removes the column with nulls from the dataset
            - 'drop-rows' : removes all the rows with the nulls values from the dataset
            - 'predict-by-sklearn' : predict values to impute nulls based on the features in the rows; used for multivariate data
            - 'impute-by-mode' : impute nulls by mode of the column values without nulls
            - 'impute-by-mode-trimmed' : the same as 'impute-by-mode', but the column is filtered from nulls,
            sorted in descending order, and top and bottom k% are removed from it. After that 'impute-by-mode' logic is applied
            - 'impute-by-mean' : impute nulls by mean of the column values without nulls
            - 'impute-by-mean-trimmed' : the same as 'impute-by-mean', but the column is filtered from nulls,
            sorted in descending order, and top and bottom k% are removed from it. After that 'impute-by-mean' logic is applied
            - 'impute-by-median' : impute nulls by median of the column values without nulls
            - 'impute-by-median-trimmed' : the same as 'impute-by-median', but the column is filtered from nulls,
            sorted in descending order, and top and bottom k% are removed from it. After that 'impute-by-median' logic is applied
    column-names: list of column names, for which the particular techniques needs to be applied

    Output:
    dataframe with processed nulls
    """
    data = input_data.copy(deep=True)

    if how == 'drop-column':
        data.drop(columns=column_names,  axis=1, inplace=True)
    elif how == 'drop-rows':
        data.dropna(subset=column_names, inplace=True)
    elif how == 'predict-by-sklearn':
        # Setting the random_state argument for reproducibility
        imputer = IterativeImputer(random_state=42)
        imputed = imputer.fit_transform(data)
        data = pd.DataFrame(imputed, columns=data.columns)
    else:
        get_impute_value = None
        if how == 'special':
            get_impute_value = decide_special_category
        elif 'impute-by-mode' in how:
            get_impute_value = find_column_mode
        elif 'impute-by-mean' in how:
            get_impute_value = find_column_mean
        elif 'impute-by-median' in how:
            get_impute_value = find_column_median

        vals = {}
        for col in column_names:
            if condition_column:
                filtered_df = data[~data[col].isnull()][[col, condition_column]].copy(deep=True)
            else:
                filtered_df = data[~data[col].isnull()][[col]].copy(deep=True)
            if 'trimmed' in how:
                k_percent = 10
                reduce_n_rows = int(filtered_df.shape[0] / 100 * k_percent)
                filtered_df.sort_values(by=[col], ascending=False, inplace=True)
                filtered_df = filtered_df[reduce_n_rows: -reduce_n_rows]

            if 'conditional' in how:
                if condition_column == 'AGEP':
                    threshold_age = 40
                    fillna_val_less = get_impute_value(filtered_df[filtered_df[condition_column] < threshold_age][col].values)
                    fillna_val_more = get_impute_value(filtered_df[filtered_df[condition_column] >= threshold_age][col].values)
                    for idx, fillna_val in enumerate(fillna_val_less, fillna_val_more):
                        sign = "<" if idx == 0 else ">="
                        print(f"Impute {col} with value {fillna_val}, where {condition_column} {sign} {threshold_age}")
                        data[col] = data.groupby(condition_column)[col].transform(lambda x: x.fillna(fillna_val))
                else:
                    mapping_dict = dict()
                    for val in filtered_df[condition_column].unique():
                        fillna_val = get_impute_value(filtered_df[filtered_df[condition_column] == val][col].values)
                        print(f"Impute {col} with value {fillna_val}, where {condition_column} == {val}")
                        mapping_dict[val] = fillna_val

                    missing_mask = data[col].isna()
                    data.loc[missing_mask, col] = data.loc[missing_mask, condition_column].map(mapping_dict)
            else:
                vals[col] = get_impute_value(filtered_df[col].values)
        if len(vals) > 0:
            print("Impute values: ", vals)
            data.fillna(value=vals, inplace=True)
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
    :param data: a pandas dataframe, in which nulls values should be simulated
    :param target_col: a column in the dataset, in which nulls should be placed
    :param condition_col: a column in the dataset based on which null location should be identified
    :param special_values: list of special values for the condition column; special_values and condition_col state the condition,
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
    """
    Description: Decides which value to designate as a special value, based on the values already in the data
    """
    if 0 not in data:
        return 0
    else:
        return max(data) + 1


def find_column_mode(data):
    result = stats.mode(data)
    return result.mode[0]


def find_column_mean(data):
    return np.mean(data)


def find_column_median(data):
    return np.median(data)
