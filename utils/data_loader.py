import pandas as pd
import numpy as np
from sys import getsizeof
from folktables import ACSDataSource

from utils.null_handler import initially_handle_nulls


def ACSDataLoader(task, state, year, without_nulls):
    '''
    Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
    We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
    Alternatively, we could have altered the configuration from here:
    https://github.com/zykls/folktables/blob/main/folktables/acs.py
    '''
    data_source = ACSDataSource(
        survey_year=year,
        horizon='1-Year',
        survey='person'
    )
    acs_data = data_source.get_data(states=state, download=True)
    X = acs_data[task.features]
    y = acs_data[task.target].apply(lambda x: int(x == 1))

    # If the task is ACSEmployment, we can optimize the file size
    print(f'Original: {int(getsizeof(X) / 1024**2)} mb')
    X_data = optimize_ACSEmployment(X)
    print(f'Optimized: {int(getsizeof(optimize_ACSEmployment(X_data)) / 1024**2)} mb\n')

    if without_nulls:
        # Encode initial nulls in the dataset as a separate category, what was proved in EDA/EDA_CA_2016.ipynb notebook
        missing = ['SCHL', 'ESP', 'MIG', 'MIL', 'DREM']
        X_data = initially_handle_nulls(X_data, missing)
        # Rechecking if there are nulls -- if the null_handler has run correctly, there should not be
        print('\nRechecking if there are nulls in X_data:')
        print(X_data.isnull().sum())

    return X_data, y


def optimize_ACSEmployment(data):
    '''
    Optimizing the dataset size by downcasting categorical columns
    '''
    categorical = ['SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data

