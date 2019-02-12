import numpy as np
import pandas as pd

from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import cohen_kappa_score, make_scorer

from preprocessing import TSFormatting, LowVarianceFeaturesRemover, preprocess_data
from classification import classify
from results_exploration import plot_avg_feature_importance

import sys


# Parameters
data_folder = './data/'
features_folder = './features/tsfresh/'
submission_folder = './submissions/benchmark/'

recompute_training = False
recompute_test = False
nb_splits = 8

perform_classification = True
perform_cross_validation = True

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
def compute_tsfresh_features(x, save_path, nb_splits=8, which_set='training'):
    print('Processing %s set...' % (which_set))
    n = x.shape[0]
    split_breaks = [int(n / nb_splits) * i for i in range(nb_splits)] + [n]
    for i in range(nb_splits):
        start = split_breaks[i]
        stop = split_breaks[i + 1]
        print('Number of rows being processed:', stop - start)
        features = extract_features(TSFormatting().transform(x.iloc[start:stop]),
                                    column_id='id', column_sort='time',
                                    default_fc_parameters=EfficientFCParameters())
        features['neuron_id'] = x.iloc[start:stop]['neuron_id']
        if (i == 0):
            features.to_csv(save_path, mode='w', header=True, index=True)
        else:
            features.to_csv(save_path, mode='a', header=False, index=True)
        del features

if recompute_training:
    compute_tsfresh_features(
        x=x_tr,
        save_path=(features_folder + 'feat_tr.csv'),
        nb_splits=nb_splits,
        which_set='training'
    )

if recompute_test:
    compute_tsfresh_features(
        x=x_te,
        save_path=(features_folder + 'feat_te.csv'),
        nb_splits=nb_splits,
        which_set='test'
    )


###############################################################################
#                                                                             #
#                             Classification part                             #
#                                                                             #
###############################################################################

if not perform_classification:
    sys.exit()


# Load features
x_tr = pd.read_csv(features_folder + 'feat_tr.csv', index_col=[0])
x_te = pd.read_csv(features_folder + 'feat_te.csv', index_col=[0])

y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])

n_tr = len(x_tr)
n_te = len(x_te)
print(n_tr, 'training samples /', n_te, 'test samples\n')


# Pre-processing
preprocessing_steps = [LowVarianceFeaturesRemover(), StandardScaler()]
x_tr, x_te, groups_tr = preprocess_data(x_tr, x_te, preprocessing_steps=preprocessing_steps)


# Classifier possibilities and parameters
est_list = [
    RandomForestClassifier(),
    XGBClassifier(),
    LGBMClassifier(),
    LogisticRegression()
]
cv_params = [
    {   # RandomForestClassifier
        'n_estimators': [200],
        'max_depth': [10],
        'class_weight': ['balanced']        
    },
    {   # XGBClassifier
        'n_estimators': [150, 200, 250],
        'max_depth': [5, 10, 15],
        'scale_pos_weight': [1/0.15, 1/0.184, 1/0.20],
        'objective': ['binary:hinge', 'binary:logistic']
    },
    {   # LGBMClassifier
        'n_estimators': [100],
        'num_leaves': [35],
        'reg_alpha': [1e-2, 1e0, 1e1, 1e2],
        'reg_lambda': [1e-2, 1e0, 1e1, 1e2],
        'max_depth': [7],
        'min_child_weight': [1e1],
        'min_child_samples': [20],
        'class_weight': ['balanced']   
    },
    {   # LogisticRegression
        'C': [1e-2, 1.0, 1e2],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced'],
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
    },
    {   # LGBMClassifier
        'n_estimators': 100,
        'num_leaves': 35,
        'reg_alpha': 10,
        'reg_lambda': 1,
        'max_depth': 7,
        'min_child_weight': 1e1,
        'min_child_samples': 20,
        'class_weight': 'balanced'   
    },
    {   # LogisticRegression
        'C': 1.0,
        'class_weight': 'balanced'
    }
]
est_idx = 3


# Classification
# TODO This try except to catch .values error should be fixed in StandardScaler
try:
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
except:
    clf = classify(
        x_tr=x_tr,
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
if plot_feature_importance:
    try:
        plot_avg_feature_importance(clf.feature_importances_, x_tr.columns)
    except:
        print('Feature importance not available\n')


# Compute submission
if compute_submission:
    clf.fit(x_tr.values, y_tr.values.ravel())
    y_te_pred = clf.predict(x_te)
    y_te_pred_df = pd.DataFrame(data=y_te_pred, columns=['TARGET'], index=(x_te.index))
    y_te_pred_df.index.name = 'ID'
    y_te_pred_df.to_csv(submission_folder + 'y_te_pred.csv', header=True, index=True)
