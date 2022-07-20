import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import COLUMN_TO_TYPE, SEED


def get_column_type(column_name):
    for column_type in COLUMN_TO_TYPE.keys():
        if column_name in COLUMN_TO_TYPE[column_type]:
            return column_type
    return None


def make_feature_df(data, categorical_columns, numerical_columns):
    """
    Return a dataset made by one-hot encoding for categorical columns and concatenate with numerical columns
    """
    feature_df = pd.get_dummies(data[categorical_columns], columns=categorical_columns)
    for col in numerical_columns:
        if col in data.columns:
            feature_df[col] = data[col]
    return feature_df


def preprocess_dataset(X_imputed, y_data, categorical_columns = COLUMN_TO_TYPE['categorical'],
                       numerical_columns = COLUMN_TO_TYPE['numerical']):
    X_features = make_feature_df(X_imputed, categorical_columns, numerical_columns)
    X_train_features, X_test_features, y_train, y_test = train_test_split(X_features, y_data, test_size=0.2, random_state=SEED)

    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    return X_train_features, y_train, X_test_features, y_test


def check_conditional_techniques(corrupted_data, target_column):
    """
    Return a dict in the next format:
    {'<key_to_group_stats>': {'RAC1P-1': (<max_value>, '<target_column>-<most_often_occurred_value>'),
               'RAC1P-2': (<max_value>, '<target_column>-<most_often_occurred_value>'),
               ...}}

    {'RAC1P': {'RAC1P-1': (2273, 'AGEP-60.0'),
               'RAC1P-2': (435, 'AGEP-19.0'),
               ...},
     'SEX': {'SEX-1': (1473, 'AGEP-19.0'), 'SEX-2': (1623, 'AGEP-60.0')}}

     'RAC1P' and 'SEX' are initial keys to group appropriate statistics.

     This function is used to check correctness of 'impute-by-(mode/mean/median)-conditional' techniques
    """
    mapping_dict = dict()
    for condition_column in ['SEX', 'RAC1P']:
        mapping_dict[condition_column] = {}
        corrupted_data_slice = corrupted_data[~corrupted_data[target_column].isnull()][[target_column, condition_column]]
        for val in corrupted_data_slice[condition_column].unique():
            counts_df = corrupted_data_slice[corrupted_data_slice[condition_column] == val][target_column].value_counts()
            mapping_dict[condition_column][f"{condition_column}-{val}"] = counts_df.iloc[0], f'{target_column}-{counts_df.index[0]}'

    return mapping_dict


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
