import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.analysis_helper import load_groups_of_interest, compute_metric
from utils.EDA_utils import plot_generic
from utils.simple_utils import preprocess_dataset, set_size
from config import *


def generate_bootstrap(features, labels, boostrap_size, with_replacement=True):
    bootstrap_index = np.random.choice(features.shape[0], size=boostrap_size, replace=with_replacement)
    bootstrap_features = pd.DataFrame(features).iloc[bootstrap_index].values
    bootstrap_labels = pd.DataFrame(labels).iloc[bootstrap_index].values
    if len(bootstrap_features) == boostrap_size:
        return bootstrap_features, bootstrap_labels
    else:
        raise ValueError('Bootstrap samples are not of the size requested')


def UQ_by_boostrap(X_train, y_train, X_test, y_test, base_model, n_estimators, boostrap_size, with_replacement=True, verbose=True):
    '''
    Quantifying uncertainty of predictive model by constructing an ensemble from boostrapped samples
    '''
    predictions = {}
    ensemble = {}
    
    for m in range(n_estimators):
        model = base_model
        X_sample, y_sample = generate_bootstrap(X_train, y_train, boostrap_size, with_replacement)
        model.fit(X_sample, y_sample)
        predictions[m] = model.predict_proba(X_test)[:, 0]
    
        if verbose:
            print(m)
            print("Train acc:", model.score(X_sample, y_sample))
            print("Val acc:", model.score(X_test, y_test))

        ensemble[m] = model
    return ensemble, predictions


def compute_label_stability(predicted_labels):
    '''
    Label stability is defined as the absolute difference between the number of times the sample is classified as 0 and 1
    If the absolute difference is large, the label is more stable
    If the difference is exactly zero then it's extremely unstable --- equally likely to be classified as 0 or 1
    '''
    count_pos = sum(predicted_labels)
    count_neg = len(predicted_labels) - count_pos
    return np.abs(count_pos - count_neg)/len(predicted_labels)


def count_prediction_stats(y_test, uq_results):
    results = pd.DataFrame(uq_results).transpose()
    means = results.mean().values
    stds = results.std().values
    iqr = sp.stats.iqr(results, axis=0)

    y_preds = np.array([int(x<0.5) for x in results.mean().values])
    accuracy = np.mean(np.array([y_preds[i] == y_test[i] for i in range(len(y_test))]))

    return y_preds, results, means, stds, iqr, accuracy


def get_per_sample_accuracy(y_test, results):
    """

    :param y_test: y test dataset
    :param results: results variable from count_prediction_stats()
    :return: per_sample_accuracy and label_stability (refer to https://www.osti.gov/servlets/purl/1527311)
    """
    per_sample_predictions = {}
    label_stability = []
    per_sample_accuracy = []
    for sample in range(len(y_test)):
        per_sample_predictions[sample] =  [int(x<0.5) for x in results[sample].values]
        label_stability.append(compute_label_stability(per_sample_predictions[sample]))

        if y_test[sample] == 1:
            acc = np.mean(per_sample_predictions[sample])
        elif y_test[sample] == 0:
            acc = 1 - np.mean(per_sample_predictions[sample])
        per_sample_accuracy.append(acc)

    return per_sample_accuracy, label_stability


def get_fairness_metrics(y_preds, test_groups, y_test):
    fairness_metrics = {}
    metrics = ['Accuracy', 'Disparate_Impact', 'Equal_Opportunity', 'Statistical_Parity_Difference']

    for metric in metrics:
        fairness_metrics[metric] = compute_metric(y_preds, y_test, test_groups, metric)

    return fairness_metrics


def quantify_uncertainty(null_scenario_name, y_data, imputed_data_dict, imputation_technique, n_estimators=200, make_plots=True):
    """
    Quantify uncertainty for the best model. Display plots for analysis if needed. Save results to a .pkl file

    :param null_scenario_name: a name of null simulation method. Just used to name a result .pkl file
    :param y_data:  y dataset, is used to create y_train_imputed, y_test_imputed
    :param imputed_data_dict: a dict where key is a name of imputation technique, value is a dataset imputed with that technique
    :param imputation_technique: a name of imputation technique to get an appropriate imputed dataset
     from imputed_data_dict to quantify uncertainty
    :param n_estimators: number of estimators in ensemble
    :param make_plots: bool, if display plots for analysis
    """
    # Prepare an imputed dataset and split it on train and test to quantify uncertainty
    X_train_imputed, X_test_imputed, y_train_imputed, y_test_imputed, test_groups = prepare_datasets(imputed_data_dict, imputation_technique, y_data)

    # Set hyper-parameters for the best model. Use hyper-parameters, which were tuned on a drop-column dataset
    ML_drop_column_results_df = pd.read_csv(os.path.join('..', 'results', null_scenario_name, f'{null_scenario_name}_ML_drop_column_results_df.csv'))
    hyperparameters_dict = eval(ML_drop_column_results_df.loc[ML_drop_column_results_df['Model_Name'] == 'DecisionTreeClassifier', 'Model_Best_Params'].iloc[0])
    decision_tree_model = DecisionTreeClassifier(criterion=hyperparameters_dict['criterion'],
                                                 max_depth=hyperparameters_dict['max_depth'],
                                                 max_features=hyperparameters_dict['max_features'])
    boostrap_size = int(0.5 * X_train_imputed.shape[0])

    # Quantify uncertainty for the bet model
    ___, uq_results = UQ_by_boostrap(X_train_imputed, y_train_imputed, X_test_imputed, y_test_imputed,
                                     decision_tree_model, n_estimators,
                                     boostrap_size, with_replacement=True, verbose=False)

    # Count metrics
    y_preds, results, means, stds, iqr, accuracy = count_prediction_stats(y_test_imputed.values, uq_results)
    per_sample_accuracy, label_stability = get_per_sample_accuracy(y_test_imputed.values, results)

    # Display plots if needed
    if make_plots:
        plot_generic(means, stds, "Mean of probability", "Standard deviation", "Probability mean vs Standard deviation")
        plot_generic(stds, label_stability, "Standard deviation", "Label stability", "Standard deviation vs Label stability")
        plot_generic(means, label_stability, "Mean", "Label stability", "Mean vs Label stability")
        plot_generic(per_sample_accuracy, stds, "Accuracy", "Standard deviation", "Accuracy vs Standard deviation")
        plot_generic(per_sample_accuracy, iqr, "Accuracy", "Inter quantile range", "Accuracy vs Inter quantile range")

    # Count and display fairness metrics
    fairness_metrics = get_fairness_metrics(y_preds, test_groups, y_test_imputed)
    print("#" * 30, "Fairness metrics", "#" * 30)
    for key in fairness_metrics.keys():
        print('\n' + '#' * 20 + f' {key} ' + '#' * 20)
        print(fairness_metrics[key])

    # Save results to a .pkl file
    save_uncertainty_results_to_file(null_scenario_name, imputation_technique, stds, iqr, accuracy, label_stability, fairness_metrics)


def prepare_datasets(imputed_data_dict, imputation_technique, y_data):
    """
    Prepare train and test imputation datasets.

    :return: train and test imputation datasets,
        and test_groups -- advantaged and disadvantaged groups for computing fairness-related metrics
    """
    X_imputed = imputed_data_dict[imputation_technique]
    # Also dropping rows from the label
    y_data_imputed = y_data.iloc[X_imputed.index].copy(deep=True)

    X_train_imputed, y_train_imputed, X_test_imputed, y_test_imputed, test_groups = preprocess_dataset(X_imputed, y_data_imputed)
    # TODO: why do we need X_val_imputed for uncertainty analysis?
    print('X_train_imputed.shape: ', X_train_imputed.shape)
    print('X_test_imputed.shape: ', X_test_imputed.shape)

    return X_train_imputed, X_test_imputed, y_train_imputed, y_test_imputed, test_groups


def save_uncertainty_results_to_file(null_scenario_name, imputation_technique, stds, iqr, accuracy, label_stability, fairness_metrics):
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

    dir_path = os.path.join('..', 'results', null_scenario_name)
    os.makedirs(dir_path, exist_ok=True)

    filename = f"{DATASET_CONFIG['state'][0]}_{DATASET_CONFIG['year']}_{null_scenario_name}_{imputation_technique.replace('-', '_')}.pkl"
    f = open(os.path.join(dir_path, filename), "wb")
    pickle.dump(metrics_to_report,f)
    f.close()


def display_result_plots(null_scenario_name, imputation_techniques):
    results = dict()
    for imputation_technique in imputation_techniques:
        filename = f"{DATASET_CONFIG['state'][0]}_{DATASET_CONFIG['year']}_{null_scenario_name}_{imputation_technique.replace('-', '_')}.pkl"
        with open(os.path.join('..', 'results', null_scenario_name, filename), 'rb') as file:
            results[imputation_technique] = pickle.load(file)

    y_metrics = ['SPD_Race', 'SPD_Sex', 'SPD_Race_Sex']
    x_metrics = ['Label_stability', 'Accuracy', 'SD']
    for x_metric in x_metrics:
        for y_metric in y_metrics:
            display_uncertainty_plot(results, x_metric, y_metric)


def display_uncertainty_plot(results, x_metric, y_metric):
    fig, ax = plt.subplots()
    set_size(15, 8, ax)

    # List of all markers -- https://matplotlib.org/stable/api/markers_api.html
    markers = ['.', 'o', '+', '*', '|', '>', '^', 'v', '1', 's', 'x', 'D', 'P', 'H']
    techniques = results.keys()
    shapes = []
    for idx, technique in enumerate(techniques):
        a = ax.scatter(results[technique][x_metric], results[technique][y_metric], label="Race", marker=markers[idx], s=100)
        shapes.append(a)

    plt.xlabel(x_metric)
    plt.ylabel("SPD_difference")
    plt.title(f'{x_metric} [{y_metric}]', fontsize=20)
    ax.legend(shapes, techniques, fontsize=12, title='Markers')

    plt.show()
