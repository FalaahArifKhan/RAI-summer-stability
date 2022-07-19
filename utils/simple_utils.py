from config import COLUMN_TO_TYPE


def get_column_type(column_name):
    for column_type in COLUMN_TO_TYPE.keys():
        if column_name in COLUMN_TO_TYPE[column_type]:
            return column_type
    return None


def check_conditional_techniques(corrupted_data, target_column):
    mapping_dict = dict()
    for condition_column in ['SEX', 'RAC1P']:
        mapping_dict[condition_column] = {}
        corrupted_data_slice = corrupted_data[~corrupted_data[target_column].isnull()][[target_column, condition_column]]
        for val in corrupted_data_slice[condition_column].unique():
            counts_df = corrupted_data_slice[corrupted_data_slice[condition_column] == val][target_column].value_counts()
            mapping_dict[condition_column][f"{condition_column}-{val}"] = counts_df.iloc[0], f'{target_column}-{counts_df.index[0]}'

    return mapping_dict
