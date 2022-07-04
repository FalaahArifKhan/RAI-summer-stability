import pandas as pd
import numpy as np
from folktables import ACSDataSource


def ACSDataLoader(task, state, year):
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
    return X, y

def optimize_ACSEmployment(data):
    '''
    Optimizing the dataset size by downcasting categorical columns
    '''
    categorical = ['SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data

