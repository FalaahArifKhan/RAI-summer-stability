import os
import pickle
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from config import BOOTSTRAP_FRACTION, DATASET_CONFIG
from utils.EDA_utils import plot_generic
from utils.analysis_helper import load_groups_of_interest, compute_metric
from utils.stability_analysis import count_prediction_stats, get_per_sample_accuracy, generate_bootstrap


class StabilityFairnessAnalyzer:
    def __init__(self, X_data_tpl, y_data_tpl, evaluation_model, imputation_technique, null_scenario_name):
        self.imputation_technique = imputation_technique
        self.null_scenario_name = null_scenario_name

        # Prepare an imputed dataset and split it on train and test to quantify uncertainty
        self.X_train_imputed, self.X_test_imputed = X_data_tpl
        self.y_train, self.y_test = y_data_tpl

        self.evaluation_model = evaluation_model
        self.test_groups = load_groups_of_interest(os.path.join('..', 'groups.json'), self.X_test_imputed)

    def measure_metrics(self, n_estimators=200, make_plots=True):
        # For computing fairness-related metrics
        boostrap_size = int(BOOTSTRAP_FRACTION * self.X_train_imputed.shape[0])

        # Quantify uncertainty for the bet model
        ___, uq_results = self.UQ_by_boostrap(n_estimators, boostrap_size, with_replacement=True, verbose=False)

        # Count metrics
        y_preds, results, means, stds, iqr, accuracy = count_prediction_stats(self.y_test.values, uq_results)
        per_sample_accuracy, label_stability = get_per_sample_accuracy(self.y_test.values, results)

        # Display plots if needed
        if make_plots:
            plot_generic(means, stds, "Mean of probability", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Probability mean vs Standard deviation")
            plot_generic(stds, label_stability, "Standard deviation", "Label stability", x_lim=0.5, y_lim=1.01, plot_title="Standard deviation vs Label stability")
            plot_generic(means, label_stability, "Mean", "Label stability", x_lim=1.01, y_lim=1.01, plot_title="Mean vs Label stability")
            plot_generic(per_sample_accuracy, stds, "Accuracy", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Accuracy vs Standard deviation")
            plot_generic(per_sample_accuracy, iqr, "Accuracy", "Inter quantile range", x_lim=1.01, y_lim=1.01, plot_title="Accuracy vs Inter quantile range")

        # Count and display fairness metrics
        fairness_metrics = self.get_fairness_metrics(y_preds, self.test_groups, self.y_test)
        print("#" * 30, "Fairness metrics", "#" * 30)
        for key in fairness_metrics.keys():
            print('\n' + '#' * 20 + f' {key} ' + '#' * 20)
            print(fairness_metrics[key])

        # Save results to a .pkl file
        self.save_metrics_to_file(stds, iqr, accuracy, label_stability, fairness_metrics)

    def UQ_by_boostrap(self, n_estimators, boostrap_size, with_replacement=True, verbose=True):
        '''
        Quantifying uncertainty of predictive model by constructing an ensemble from boostrapped samples
        '''
        predictions = {}
        ensemble = {}

        for m in range(n_estimators):
            model = self.evaluation_model
            X_sample, y_sample = generate_bootstrap(self.X_train_imputed, self.y_train, boostrap_size, with_replacement)
            model.fit(X_sample, y_sample)
            predictions[m] = model.predict_proba(self.X_test_imputed)[:, 0]

            if verbose:
                print(m)
                print("Train acc:", model.score(X_sample, y_sample))
                print("Val acc:", model.score(self.X_test_imputed, self.y_test))

            ensemble[m] = model
        return ensemble, predictions

    def get_fairness_metrics(self, y_preds, test_groups, y_test):
        fairness_metrics = {}
        metrics = ['Accuracy', 'Disparate_Impact', 'Equal_Opportunity', 'Statistical_Parity_Difference']

        for metric in metrics:
            fairness_metrics[metric] = compute_metric(y_preds, y_test, test_groups, metric)

        return fairness_metrics

    def save_metrics_to_file(self, stds, iqr, accuracy, label_stability, fairness_metrics):
        metrics_to_report = {}
        metrics_to_report['Accuracy'] = accuracy
        metrics_to_report['Label_stability'] = [np.mean(label_stability)]
        metrics_to_report['SD'] = [np.mean(stds)]
        metrics_to_report['IQR'] = [np.mean(iqr)]
        metrics_to_report['SPD_Race'] = [fairness_metrics['Statistical_Parity_Difference']['Race'].loc['Statistical_Parity_Difference']]
        metrics_to_report['SPD_Sex'] = [fairness_metrics['Statistical_Parity_Difference']['Sex'].loc['Statistical_Parity_Difference']]
        metrics_to_report['SPD_Race_Sex'] = [fairness_metrics['Statistical_Parity_Difference']['Race_Sex'].loc['Statistical_Parity_Difference']]
        metrics_to_report['EO_Race'] = [fairness_metrics['Equal_Opportunity']['Race'].loc['Equal_Opportunity']]
        metrics_to_report['EO_Sex'] = [fairness_metrics['Equal_Opportunity']['Sex'].loc['Equal_Opportunity']]
        metrics_to_report['EO_Race_Sex'] = [fairness_metrics['Equal_Opportunity']['Race_Sex'].loc['Equal_Opportunity']]
        pd.DataFrame(metrics_to_report)

        dir_path = os.path.join('..', 'results', self.null_scenario_name)
        os.makedirs(dir_path, exist_ok=True)

        filename = f"{DATASET_CONFIG['state'][0]}_{DATASET_CONFIG['year']}_{self.null_scenario_name}_{self.imputation_technique.replace('-', '_')}.pkl"
        f = open(os.path.join(dir_path, filename), "wb")
        pickle.dump(metrics_to_report,f)
        f.close()