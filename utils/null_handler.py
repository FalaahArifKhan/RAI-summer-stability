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


def decide_special_category(data):
    '''
    Description: Decides which value to designate as a special value, based on the values already in the data
    '''
    if 0 not in data:
        return 0
    else:
        return max(data)+1