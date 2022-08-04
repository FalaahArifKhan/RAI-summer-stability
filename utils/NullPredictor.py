import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utils.null_helpers import *


class NullPredictor():
    def __init__(self, base_classifier, input_columns, target_columns, categorical_columns, numerical_columns):
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.categorical_columns = [x for x in categorical_columns if x in self.input_columns]
        self.numerical_columns = [x for x in numerical_columns if x in self.input_columns]

        self.target_transformer = {}
        self.base_model = {}
        self.fitted_model = {}

        for col in self.target_columns:
            column_type = 'categorical' if col in categorical_columns else 'numerical'

            # We will need to binarize categorical target columns
            self.target_transformer[col] = LabelEncoder() if column_type == 'categorical' else None

            # self.base_model[col] = base_classifier(column_type)
            self.base_model[col] = base_classifier[column_type]
            self.fitted_model[col] = None

    def fit(self, data_with_nulls, y=None):
        # Fit only on rows without nulls
        data = data_with_nulls.dropna(inplace=False)
        # print("Fitting on ", data.shape[0], " non-null values")
        for col in self.target_columns:
            X = data[self.input_columns]

            # Binarizing categorical targets before fitting
            if self.target_transformer[col] != None:
                y = self.target_transformer[col].fit_transform(data[col])
            else:
                y = data[col]

            encoder = ColumnTransformer(transformers=[
                ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_columns),
                ('numerical_features', StandardScaler(), self.numerical_columns)])
            pipeline = Pipeline([('features', encoder), ('learner', self.base_model[col])])
            # print(col, self.base_model[col])

            model = pipeline.fit(X, y)
            self.fitted_model[col] = model

    def transform(self, X, y=None):
        # Transform only those rows with nulls
        data = X.copy(deep=True)

        for col in self.target_columns:
            if self.fitted_model[col] is None:
                raise ValueError("Call fit before calling transform!")
                return data

            null_idx = np.where(data[col].isnull())[0]
            # print("Transforming ", len(null_idx), " nulls")
            X_test = data[self.input_columns].iloc[null_idx]

            predicted = self.fitted_model[col].predict(X_test)

            # Inverse transforming binary targets back into categories
            if self.target_transformer[col] != None:
                predicted = self.target_transformer[col].inverse_transform(predicted)

            data[col].iloc[null_idx] = predicted.astype('int')

        return data

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)
