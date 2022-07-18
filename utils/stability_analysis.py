import numpy as np
import pandas as pd
import scipy as sp

def generate_bootstrap(features, labels, boostrap_size, with_replacement=True):
    bootstrap_index = np.random.choice(features.shape[0], size=boostrap_size, replace=with_replacement)
    bootstrap_features = pd.DataFrame(features).loc[bootstrap_index].values
    bootstrap_labels = pd.DataFrame(labels).loc[bootstrap_index].values
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
        ensemble[m] = base_model
        X_sample, y_sample = generate_bootstrap(X_train, y_train, boostrap_size, with_replacement)
        ensemble[m].fit(X_sample, y_sample)
        predictions[m] = ensemble[m].predict_proba(X_test)[:,0]
    
        if verbose:
            print(m)
            print("Train acc:", ensemble[m].score(X_sample, y_sample))
            print("Val acc:", ensemble[m].score(X_test, y_test))
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

