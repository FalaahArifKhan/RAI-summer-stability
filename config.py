column_to_type = {
    "categorical": ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P'],
    "numerical": ['SCHL', 'AGEP']
}

# Config with simulated scenarios
simulated_scenarios_dict = {
    # "Optional" Type of Nulls
    'MAR': {
        'special_values': [2, 3, 4],
        'condition_col': 'MAR',
        'target_col': 'MAR',
        'fraction': 0.3
    },
    # "Not Applicable" Type of Nulls
    'MIL': {
        'special_values': [i for i in range(0, 17)],
        'condition_col': 'AGEP',
        'target_col': 'MIL',
        'fraction': 0.3
    },
    'SCHL': {
        'special_values': [i for i in range(0, 3)],
        'condition_col': 'AGEP',
        'target_col': 'SCHL',
        'fraction': 0.3
    },
    'ESP': {
        'special_values': [0],
        'condition_col': 'ESP',
        'target_col': 'ESP',
        'fraction': 0.3
    },
    'MIG': {
        'special_values': [0],
        'condition_col': 'AGEP',
        'target_col': 'MIG',
        'fraction': 0.3
    },
    'DREM_not_applic': {
        'special_values': [i for i in range(0, 5)],
        'condition_col': 'AGEP',
        'target_col': 'DREM',
        'fraction': 0.3
    },
    # "Unknown" Type of Nulls
    'AGEP': {
        'special_values': (8, 10, 11, 12, 15),
        'condition_col': 'RELP',
        'target_col': 'AGEP',
        'fraction': 0.3
    },
    'NATIVITY': {
        'special_values': (11, 12, 15),
        'condition_col': 'RELP',
        'target_col': 'NATIVITY',
        'fraction': 0.3
    },
    # "Avoided" Type of Nulls
    'DIS': {
        'special_values': [True],
        'condition_col': 'DIS',
        'target_col': 'DIS',
        'fraction': 0.3
    },
    'DEAR': {
        'special_values': [True],
        'condition_col': 'DEAR',
        'target_col': 'DEAR',
        'fraction': 0.3
    },
    'DEYE': {
        'special_values': [True],
        'condition_col': 'DEYE',
        'target_col': 'DEYE',
        'fraction': 0.3
    },
    'DREM_avoided': {
        'special_values': [True],
        'condition_col': 'DREM',
        'target_col': 'DREM',
        'fraction': 0.3
    },
    # "Special" Type of Nulls
    'SEX': {
        'special_values': [1],
        'condition_col': 'SEX',
        'target_col': 'SEX',
        'fraction': 0.11
    },
    'RAC1P': {
        'special_values': [9],
        'condition_col': 'RAC1P',
        'target_col': 'RAC1P',
        'fraction': 0.3
    },
}