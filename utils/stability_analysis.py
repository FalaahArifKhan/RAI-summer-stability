import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.analysis_helper import load_groups_of_interest, compute_metric
from utils.EDA_utils import plot_generic
from utils.simple_utils import preprocess_dataset
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
    iqr = sp.stats.iqr(results,axis=0)

    y_preds = np.array([int(x<0.5) for x in results.mean().values])
    accuracy = np.mean(np.array([y_preds[i] == y_test[i] for i in range(len(y_test))]))

    return y_preds, results, means, stds, iqr, accuracy


def get_per_sample_accuracy(y_test, results):
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


def quantify_uncertainty(target_column, y_data, imputed_data_dict, imputation_technique, n_estimators=50, make_plots=True):
    X_train_imputed, X_test_imputed, y_train_imputed, y_test_imputed, test_groups = prepare_datasets(imputed_data_dict, imputation_technique, y_data)

    decision_tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features=0.6)
    boostrap_size = int(0.5 * X_train_imputed.shape[0])

    ___, uq_results = UQ_by_boostrap(X_train_imputed, y_train_imputed, X_test_imputed, y_test_imputed,
                                     decision_tree_model, n_estimators,
                                     boostrap_size, with_replacement=True, verbose=False)

    y_preds, results, means, stds, iqr, accuracy = count_prediction_stats(y_test_imputed.values, uq_results)
    per_sample_accuracy, label_stability = get_per_sample_accuracy(y_test_imputed.values, results)

    if make_plots:
        plot_generic(means, stds, "Mean of probability", "Standard deviation", "Probability mean vs Standard deviation")
        plot_generic(stds, label_stability, "Standard deviation", "Label stability", "Standard deviation vs Label stability")
        plot_generic(means, label_stability, "Mean", "Label stability", "Mean vs Label stability")
        plot_generic(per_sample_accuracy, stds, "Accuracy", "Standard deviation", "Accuracy vs Standard deviation")
        plot_generic(per_sample_accuracy, iqr, "Accuracy", "Inter quantile range", "Accuracy vs Inter quantile range")

    fairness_metrics = get_fairness_metrics(y_preds, test_groups, y_test_imputed)
    print("#" * 30, "Fairness metrics", "#" * 30)
    for key in fairness_metrics.keys():
        print('\n' + '#' * 20 + f' {key} ' + '#' * 20)
        print(fairness_metrics[key])

    save_uncertainty_results_to_file(target_column, imputation_technique, stds, iqr, accuracy, label_stability, fairness_metrics)


def prepare_datasets(imputed_data_dict, imputation_technique, y_data):
    X_imputed = imputed_data_dict[imputation_technique]

    # Also dropping rows from the label
    y_data_imputed = y_data.iloc[X_imputed.index].copy(deep=True)

    # For computing fairness-related metrics
    _, X_test, _, _ = train_test_split(X_imputed, y_data_imputed, test_size=0.2, random_state=SEED)
    test_groups = load_groups_of_interest(os.path.join('..', 'groups.json'), X_test)

    X_train_imputed_, y_train_imputed_, X_test_imputed, y_test_imputed = preprocess_dataset(X_imputed, y_data_imputed)
    # TODO: why do we need X_val_imputed for uncertainty analysis?
    X_train_imputed, X_val_imputed, y_train_imputed, y_val_imputed = train_test_split(X_train_imputed_, y_train_imputed_,
                                                                                      test_size=0.25, random_state=SEED)
    print('X_train_imputed.shape: ', X_train_imputed.shape)
    print('X_val_imputed.shape: ', X_val_imputed.shape)
    print('X_test_imputed.shape: ', X_test_imputed.shape)

    return X_train_imputed, X_test_imputed, y_train_imputed, y_test_imputed, test_groups


def save_uncertainty_results_to_file(target_column, imputation_technique, stds, iqr, accuracy, label_stability, fairness_metrics):
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

    dir_path = os.path.join('..', 'results', f'{target_column}_column')
    os.makedirs(dir_path)

    filename = f"{DATASET_CONFIG['state'][0]}_{DATASET_CONFIG['year']}_{target_column}_{imputation_technique.replace('-', '_')}.pkl"
    f = open(os.path.join(dir_path, filename), "wb")
    pickle.dump(metrics_to_report,f)
    f.close()
