import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, accuracy_score, f1_score

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

            self.base_model[col] = base_classifier(column_type)
            self.fitted_model[col] = None

    def fit(self, data_with_nulls, y=None):
        # Fit only on rows without nulls
        data = data_with_nulls.dropna(inplace=False)
        # print("Fitting on ", data.shape[0], " non-null values")
        for col in self.target_columns:
            X = data[self.input_columns]

            # Binarizing categorical targets before fitting
            if self.target_transformer[col] != None:
                print('data[col].unique(): ', data[col].unique())
                self.y = self.target_transformer[col].fit_transform(data[col])
                print('self.y.unique(): ', self.y[:50])
            else:
                self.y = data[col]

            # encoder = ColumnTransformer(transformers=[
            #     ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_columns),
            #     # ('numerical_features', StandardScaler(), self.numerical_columns)
            # ])
            # pipeline = Pipeline([('features', encoder), ('learner', self.base_model[col])])
            pipeline = Pipeline([('learner', self.base_model[col])])
            # print(col, self.base_model[col])

            print('X train shape: ', X.shape)
            model = pipeline.fit(X, self.y)
            print("Fit score: ", pipeline.score(X, self.y))
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
            print('X X_test shape: ', X_test.shape)
            # y_test = data[col][data[col].isnull()]

            # X_test = self.encoder.transform(X_test)
            # print('X_test.head(): ', X_test.head())
            predicted = self.fitted_model[col].predict(X_test)

            # Inverse transforming binary targets back into categories
            if self.target_transformer[col] != None:
                predicted = self.target_transformer[col].inverse_transform(predicted)
                print('Inverse transform for predicted values')

            print('unique predicted: ', np.unique(predicted))
            if isinstance(y, pd.Series):
                try:
                    print('X_test.shape: ', X_test.shape)
                    print('y.shape: ', y.shape)
                    print("Prediction score1: ", self.fitted_model[col].score(X_test, y))
                    if self.target_transformer[col]:
                        print("Prediction accuracy score: ", accuracy_score(y, predicted.round()))
                        print("Prediction f1 score: ", f1_score(y, predicted.round(), average='macro'))
                    else:
                        print("Prediction RMSE score: ", mean_squared_error(y, predicted.round(), squared=False))
                        print("Prediction MAE score: ", mean_absolute_error(y, predicted.round()))
                except Exception as err:
                    print(err)
            # print("Predicted: ", predicted)

            # data[col].iloc[null_idx] = predicted.astype('int')
            data[col].iloc[null_idx] = predicted
            data[col] = data[col].astype(int)

        return data

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)
