import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from config import SEED


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


def decide_special_category(data):
    """
    Description: Decides which value to designate as a special value, based on the values already in the data (array)
    """
    data_type = data.dtype
    # If not numerical, simply set the special value to "special"
    try:
        # If data is numerical
        if 0 not in data:
            return 0
        else:
            return max(data) + 1
    except:
        print("Data is not numerical, assigning string category")
        return data_type("Special")


def find_column_mode(data):
    result = stats.mode(data)
    return result.mode[0]


def find_column_mean(data):
    return np.mean(data).round()


def find_column_median(data):
    return np.median(data).round()


def base_regressor(column_type):
    if column_type == 'numerical':
        model = LinearRegression()
    elif column_type == 'categorical':
        model = LogisticRegression()
    else:
        raise ValueError(
                "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model


def base_knn(column_type, n_neighbors):
    if column_type == 'numerical':
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif column_type == 'categorical':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError(
                "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model


def base_random_forest(column_type, n_estimators, max_depth, class_weight=None, min_samples_leaf=None, oob_score=None):
    if column_type == 'numerical':
        model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, random_state = SEED)
    elif column_type == 'categorical':
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state = SEED,
                                       class_weight=class_weight,
                                       min_samples_leaf=min_samples_leaf,
                                       oob_score=oob_score)
    else:
        raise ValueError(
            "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model
