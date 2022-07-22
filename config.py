from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from folktables import ACSEmployment


SEED=10
BOOTSTRAP_FRACTION = 1.0
DATASET_CONFIG = {
    'state': ["NY"],
    'year': '2018',
    'task': ACSEmployment
}

COLUMN_TO_TYPE = {
    "categorical": ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC'],
    "numerical": ['SCHL', 'AGEP']
}

MODELS_CONFIG = [
    {
        'model_name': 'LogisticRegression',
        'model': LogisticRegression(random_state=SEED),
        'params': {
            'penalty': ['none', 'l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': range(50, 251, 50),
        }
    },
    {
        'model_name': 'DecisionTreeClassifier',
        'model': DecisionTreeClassifier(random_state=SEED),
        'params': {
            "max_depth": [2, 3, 4, 6, 10],
            "max_features": [0.6, 'sqrt'],
            "criterion": ["gini", "entropy"]
        }
    },
    {
        'model_name': 'XGBClassifier',
        'model': XGBClassifier(random_state=SEED, verbosity = 0),
        'params': {
            'learning_rate': [0.1],
            'n_estimators': [200, 300],
            'max_depth': range(3, 10, 2),
            'objective':  ['binary:logistic'],
        }
    }
]

# Config with simulated scenarios
SIMULATED_SCENARIOS_DICT = {
    # "Optional" Type of Nulls
    'Optional_MAR': {
        'special_values': [2, 3, 4],
        'condition_col': 'MAR',
        'target_col': 'MAR',
        'fraction': 0.3
    },
    # "Not Applicable" Type of Nulls
    # 'Not_Applicable_MIL': {
    #     'special_values': [i for i in range(0, 17)],
    #     'condition_col': 'AGEP',
    #     'target_col': 'MIL',
    #     'fraction': 0.3
    # },
    'Not_Applicable_SCHL': {
        'special_values': [i for i in range(0, 3)],
        'condition_col': 'AGEP',
        'target_col': 'SCHL',
        'fraction': 0.3
    },
    # 'Not_Applicable_ESP': {
    #     'special_values': [0],
    #     'condition_col': 'ESP',
    #     'target_col': 'ESP',
    #     'fraction': 0.3
    # },
    # 'Not_Applicable_MIG': {
    #     'special_values': [0],
    #     'condition_col': 'AGEP',
    #     'target_col': 'MIG',
    #     'fraction': 0.3
    # },
    # 'Not_Applicable_DREM': {
    #     'special_values': [i for i in range(0, 5)],
    #     'condition_col': 'AGEP',
    #     'target_col': 'DREM',
    #     'fraction': 0.3
    # },
    # "Unknown" Type of Nulls
    'Unknown_AGEP': {
        'special_values': (8, 10, 11, 12, 15),
        'condition_col': 'RELP',
        'target_col': 'AGEP',
        'fraction': 0.3
    },
    # 'Unknown_NATIVITY': {
    #     'special_values': (11, 12, 15),
    #     'condition_col': 'RELP',
    #     'target_col': 'NATIVITY',
    #     'fraction': 0.3
    # },
    # "Avoided" Type of Nulls
    'Avoided_DIS': {
        'special_values': [True],
        'condition_col': 'DIS',
        'target_col': 'DIS',
        'fraction': 0.3
    },
    # 'Avoided_DEAR': {
    #     'special_values': [True],
    #     'condition_col': 'DEAR',
    #     'target_col': 'DEAR',
    #     'fraction': 0.3
    # },
    # 'Avoided_DEYE': {
    #     'special_values': [True],
    #     'condition_col': 'DEYE',
    #     'target_col': 'DEYE',
    #     'fraction': 0.3
    # },
    # 'Avoided_DREM': {
    #     'special_values': [True],
    #     'condition_col': 'DREM',
    #     'target_col': 'DREM',
    #     'fraction': 0.3
    # },
    # "Special" Type of Nulls
    'Special_SEX': {
        'special_values': [1],
        'condition_col': 'SEX',
        'target_col': 'SEX',
        'fraction': 0.11
    },
    # 'Special_RAC1P': {
    #     'special_values': [9],
    #     'condition_col': 'RAC1P',
    #     'target_col': 'RAC1P',
    #     'fraction': 0.3
    # },
}