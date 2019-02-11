# TODO Keep the index in features saving

import numpy as np
import pandas as pd

from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import cohen_kappa_score, make_scorer

from preprocessing import TSFormatting, LowVarianceFeaturesRemover, preprocess_data

import matplotlib.pyplot as plt
import sys


# Parameters
data_folder = './data/'
features_folder = './features/tsfresh/'
submission_folder = './submissions/benchmark/'

recompute_training = False
recompute_test = False
nb_splits = 8

perform_classification = True
perform_cross_validation = False

plot_feature_importance = True

compute_submission = False


# For now, disable recomputing features and performing classification
# at the same time to prevent stupid mistakes
assert((recompute_test | recompute_training) ^ perform_classification)



###############################################################################
#                                                                             #
#                             Pre-processing part                             #
#                                                                             #
###############################################################################

# Load data if pre-processing is required
if (recompute_test | recompute_training):
    x_tr = pd.read_csv(data_folder + 'training.csv', index_col=[0])
    x_te = pd.read_csv(data_folder + 'input_test.csv', index_col=[0])

    y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])

    n_tr = len(x_tr)
    n_te = len(x_te)
    print(n_tr, 'training samples /', n_te, 'test samples')


# Features computation
if recompute_training:
    print('Processing training set...')
    split_breaks = [int(n_tr / nb_splits) * i for i in range(nb_splits)] + [n_tr]
    for i in range(nb_splits):
        start = split_breaks[i]
        stop = split_breaks[i + 1]
        print('Number of rows processed:', stop - start)
        features_tr = extract_features(TSFormatting().transform(x_tr.iloc[start:stop]),
                                        column_id='id', column_sort='time',
                                        default_fc_parameters=EfficientFCParameters())
        features_tr['neuron_id'] = x_tr.iloc[start:stop]['neuron_id']
        if (i == 0):
            features_tr.to_csv(features_folder + 'feat_tr.csv', mode='w', header=True, index=False)
        else:
            features_tr.to_csv(features_folder + 'feat_tr.csv', mode='a', header=False, index=False)
        del features_tr

if recompute_test:
    print('Processing test set...')
    split_breaks = [int(n_te / nb_splits) * i for i in range(nb_splits)] + [n_te]
    for i in range(nb_splits):
        start = split_breaks[i]
        stop = split_breaks[i + 1]
        print('Number of rows processed:', stop - start)
        features_te = extract_features(TSFormatting().transform(x_te.iloc[start:stop]),
                                        column_id='id', column_sort='time',
                                        default_fc_parameters=EfficientFCParameters())
        features_te['neuron_id'] = x_te.iloc[start:stop]['neuron_id']
        if (i == 0):
            features_te.to_csv(features_folder + 'feat_te.csv', mode='w', header=True, index=False)
        else:
            features_te.to_csv(features_folder + 'feat_te.csv', mode='a', header=False, index=False)
        del features_te



###############################################################################
#                                                                             #
#                             Classification part                             #
#                                                                             #
###############################################################################

if not perform_classification:
    sys.exit()


# Load features
x_tr = pd.read_csv(features_folder + 'feat_tr.csv')
x_te = pd.read_csv(features_folder + 'feat_te.csv')

y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])

n_tr = len(x_tr)
n_te = len(x_te)
print(n_tr, 'training samples /', n_te, 'test samples\n')


# Pre-processing
preprocessing_steps = [LowVarianceFeaturesRemover(), StandardScaler()]
x_tr, x_te, groups_tr = preprocess_data(x_tr, x_te, preprocessing_steps=None)


# Classifier possibilities
est_list = [
    RandomForestClassifier(),
    XGBClassifier()
]
cv_params = [
    {   # RandomForestClassifier
        'n_estimators': [150, 200, 250],
        'max_depth': [10, 11, 12, 13],
        'class_weight': ['balanced']        
    },
    {   # XGBClassifier
        'n_estimators': [150, 200, 250],
        'max_depth': [5, 10, 15],
        'scale_pos_weight': [1/0.15, 1/0.184, 1/0.20],
        'objective': ['binary:hinge', 'binary:logistic']
    }
]
best_params = [
    {   # RandomForestClassifier
        'n_estimators': 200,
        'max_depth': 10,
        'class_weight': 'balanced'        
    },
    {   # XGBClassifier
        'n_estimators': 200,
        'max_depth': 2,
        'scale_pos_weight': 1/0.184,
        'objective': 'binary:logistic'       
    }
]
est_idx = 0


# Classification wrapper
def classify(x_tr, y_tr, groups_tr, est, est_params=None,
             perform_cross_validation=False, cv_params=None,
             random_state=None):
    '''
    Classification wrapper, handling cross-validation and fitting
    '''
    assert((est_params is not None)
           | (perform_cross_validation & (cv_params is not None)))

    splitter = GroupShuffleSplit(n_splits=5, test_size=0.33,
                                 random_state=random_state)

    if perform_cross_validation:
        print('Cross-validating the following estimator:\n', est, '\n',
              'on the following parameters: %s' % list(cv_params.keys()), '\n')
        gscv = GridSearchCV(est, cv_params, verbose=1,
                            scoring=make_scorer(cohen_kappa_score),
                            cv=list(splitter.split(x_tr, y_tr, groups_tr)))
        gscv.fit(x_tr, y_tr)
        print('Best parameters: %r' % (gscv.best_params_))
        means = gscv.cv_results_['mean_test_score']
        stds = gscv.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print('\n')
        clf = gscv.best_estimator_

    else:
        train_idx, test_idx = next(splitter.split(x_tr, y_tr, groups_tr))

        X_train = x_tr[train_idx]           # Overloading the notations is
        X_test = x_tr[test_idx]             # not ideal, and should be avoided

        y_train = y_tr[train_idx]
        y_test = y_tr[test_idx]

        clf = est.set_params(**est_params)
        print('Fitting the following classifier:\n', clf, '\n')

        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # ROC curve
        from sklearn.metrics import roc_curve, auc
        y_train_score = clf.predict_proba(X_train)[:, 1]
        y_test_score = clf.predict_proba(X_test)[:, 1]

        print('Training score:', cohen_kappa_score(y_train, y_train_pred))
        print('Test score:', cohen_kappa_score(y_test, y_test_pred))

        print('Mean training prediction:', np.mean(y_train_pred))
        print('Mean test prediction:', np.mean(y_test_pred))

        print('\n')

    return(clf)

clf = classify(
    x_tr=x_tr.values,
    y_tr=y_tr.values.ravel(),
    groups_tr=groups_tr.values,
    est=est_list[est_idx],
    est_params=best_params[est_idx],
    perform_cross_validation=perform_cross_validation,
    cv_params=cv_params[est_idx],
    random_state=42
)
print(clf)


# Feature importance
def plot_avg_feature_importance(importances, feature_names):
    if len(importances.shape) == 1:
        importances = importances.reshape(1, -1)
    mean_importances = np.mean(importances, axis=0)
    e = np.argsort(mean_importances)[::-1][:40]
    std_importances = np.std(importances, axis=0)
    fig, ax = plt.subplots(figsize=(13,13))
    ax.barh(np.arange(len([feature_names[i] for i in e])), mean_importances[e],
            xerr=std_importances[e], align='center')
    ax.set_yticks(np.arange(len([feature_names[i] for i in e])))
    ax.set_yticklabels([feature_names[i] for i in e])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature importance and variability')
    plt.tight_layout()
    plt.show()

if plot_feature_importance:
    try:
        plot_avg_feature_importance(clf.feature_importances_, x_tr.columns)
    except:
        print('Feature importance not available\n')


# Compute submission
if compute_submission:
    clf.fit(x_tr.values, y_tr.values.ravel())
    y_te_pred = clf.predict(x_te)
    y_te_pred_df = pd.DataFrame(data=y_te_pred, columns=['TARGET'], index=(16635 + x_te.index))
    y_te_pred_df.index.name = 'ID'
    y_te_pred_df.to_csv(submission_folder + 'y_te_pred.csv', header=True, index=True)