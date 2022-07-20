import pandas as pd
import numpy as np
import seaborn as sns

from scipy import stats
from sklearn.experimental import enable_iterative_imputer # Required for IterativeImputer
from sklearn.impute import IterativeImputer

from utils.EDA_utils import imputed_nulls_analysis
from config import SEED


def impute_df_with_all_techniques(real_data, corrupted_data, target_column, column_type, enable_plots=True):
    """
    Impute target_column in corrupted_data with appropriate techniques.

    :param real_data: an original dataset without nulls
    :param corrupted_data: a corrupted dataset with nulls, created based on one of null scenarios
    :param target_column: a column in corrupted_data, which has simulated nulls
    :param column_type: categorical or numerical, a type of target_column
    :param enable_plots: bool, if display plots for analysis or not

    :return: a dict, where a key is a name of an imputation technique, value is an imputed datasets with respective techniques
    """
    if column_type == "categorical":
        how_to_list = ["drop-column", "drop-rows", "predict-by-sklearn",
                       "impute-by-mode", "impute-by-mode-trimmed", "impute-by-mode-conditional"]
    elif column_type == "numerical":
        how_to_list = ["drop-column", "drop-rows", "predict-by-sklearn",
                       "impute-by-mean", "impute-by-mean-trimmed", "impute-by-mean-conditional",
                       "impute-by-median", "impute-by-median-trimmed", "impute-by-median-conditional"]
    else:
        raise ValueError("Incorrect input column_type. It must be in ('categorical', 'numerical')")

    # Set style for seaborn plots
    sns.set_style("darkgrid")

    # Save a result imputed dataset in imputed_data_dict for each imputation technique
    imputed_data_dict = dict()
    for how_to in how_to_list:
        print("\n" * 4, "#" * 15, f" Impute {target_column} column with {how_to} technique ", "#" * 15)
        imputed_data = None
        if 'conditional' in how_to:
            for condition_column in ["SEX", "RAC1P"]:
                # When condition_column == target_column, imputation based on subgroups does not make sense
                if condition_column == target_column:
                    continue
                imputed_data = handle_df_nulls(corrupted_data,
                                               how_to,
                                               condition_column=condition_column,
                                               column_names=[target_column])
                imputed_data_dict[f'{how_to}_{condition_column}'] = imputed_data

        else:
            imputed_data = handle_df_nulls(corrupted_data,
                                           how_to,
                                           condition_column=None,
                                           column_names=[target_column])
            imputed_data_dict[how_to] = imputed_data

        # Make plots for other techniques except "drop-column", since we dropped the column based on this technique
        if enable_plots and how_to != "drop-column":
            imputed_nulls_analysis(real_data, imputed_data, corrupted_data, target_col=target_column)

    return imputed_data_dict


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
        - 'impute-by-(mode/mean/median)-conditional' : the same as 'impute-by-(mode/mean/median)',
        but (mode/mean/median) is counted for each group and each group is imputed with this appropriate (mode/mean/median).
        Groups are created based on split of a dataset by RAC1P or SEX
    column-names: list of column names, for which the particular techniques needs to be applied

    Output:
    a dataframe with processed nulls
    """
    data = input_data.copy(deep=True)

    if how == 'drop-column':
        data.drop(columns=column_names,  axis=1, inplace=True)
    elif how == 'drop-rows':
        data.dropna(subset=column_names, inplace=True)
    elif how == 'predict-by-sklearn':
        if len(column_names) > 1:
            print(f"\n\nERROR: {how} technique does not work with more than one column.\n\n")
            return data

        # Setting the random_state argument for reproducibility
        imputer = IterativeImputer(random_state=SEED,
                                   min_value=input_data[column_names[0]].min(),
                                   max_value=input_data[column_names[0]].max())
        imputed = imputer.fit_transform(data)
        data = pd.DataFrame(imputed, columns=data.columns)
        data[column_names] = data[column_names].round()
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

        if 'conditional' in how:
            data = apply_conditional_technique(data, column_names, condition_column, how, get_impute_value)
        else:
            vals = {}
            for col in column_names:
                filtered_df = data[~data[col].isnull()][[col]].copy(deep=True)
                if 'trimmed' in how:
                    k_percent = 10
                    reduce_n_rows = int(filtered_df.shape[0] / 100 * k_percent)
                    filtered_df.sort_values(by=[col], ascending=False, inplace=True)
                    filtered_df = filtered_df[reduce_n_rows: -reduce_n_rows]

                vals[col] = get_impute_value(filtered_df[col].values)
            print("Impute values: ", vals)
            data.fillna(value=vals, inplace=True)
    return data


def apply_conditional_technique(data, column_names, condition_column, how, get_impute_value):
    """
    Function is used in handle_df_nulls() for 'impute-by-(mode/mean/median)-conditional' imputation techniques.
    It imputes nulls with mean/mode/median in the input dataset for each group.
    Groups are created based on split of a dataset by condition_column (RAC1P or SEX).

    :param data: a dataset for imputation
    :param column_names: column names to impute
    :param condition_column: a conditional column based on which the dataset is split on groups.
     Data in each of these groups is imputed with appropriate mean/mode/median.
     In general RAC1P or SEX, but it can be any column, except those, which are in column_names.
    :param how: a name of imputation technique
    :param get_impute_value: a function like find_column_mean or find_column_mode etc.,
    which is used to get values for the nulls imputation

    :return: a dataframe with processed nulls
    """
    for col in column_names:
        filtered_df = data[~data[col].isnull()][[col, condition_column]].copy(deep=True)
        if col == condition_column:
            print(f"\n\n\nERROR: a target column from column_names list can not be equal to conditional column. "
                  f"Skip {how} technique for {col} column\n\n\n")
            continue

        # TODO: use AGEP_categorical instead of this if-statement
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
    return data


def initially_handle_nulls(X_data, missing):
    handle_nulls = {
        'special': missing,
    }
    # Checking dataset shape before handling nulls
    print("Dataset shape before handling nulls: ", X_data.shape)

    for how_to in handle_nulls.keys():
        X_data = handle_df_nulls(X_data, how_to, handle_nulls[how_to])
    # Checking dataset shape after handling nulls
    print("Dataset shape after handling nulls: ", X_data.shape)
    return X_data


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


def simulate_scenario(data, simulated_scenario_dict):
    return nulls_simulator(data,
                           simulated_scenario_dict['target_col'],
                           simulated_scenario_dict['condition_col'],
                           simulated_scenario_dict['special_values'],
                           simulated_scenario_dict['fraction'])


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
    return np.mean(data).round()


def find_column_median(data):
    return np.median(data).round()
