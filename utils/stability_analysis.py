import os
import numpy as np
import pandas as pd
import scipy as sp

from utils.analysis_helper import load_groups_of_interest, compute_metric


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
