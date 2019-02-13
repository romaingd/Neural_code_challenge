import numpy as np
import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocessing import TSFormatting, LowVarianceFeaturesRemover, preprocess_data, CenterScaler
from classification import classify
from results_exploration import plot_avg_feature_importance

import sys


# Parameters
data_folder = './data/'
isi_folder = './features/isi/'
features_folder = './features/tsfresh/'
submission_folder = './submissions/benchmark/'

recompute_training = False
recompute_test = False
nb_splits = 8

perform_classification = True
perform_cross_validation = False

use_preprocessing = True
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

# Load ISI data if pre-processing is required
if (recompute_test | recompute_training):
    x_tr = pd.read_csv(isi_folder + 'feat_tr.csv', index_col=[0])
    x_te = pd.read_csv(isi_folder + 'feat_te.csv', index_col=[0])

    y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])


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


# Pre-processing
if use_preprocessing:
    preprocessing_steps = [LowVarianceFeaturesRemover(), CenterScaler()]
else:
    preprocessing_steps = None
x_tr, x_te, groups_tr = preprocess_data(x_tr, x_te, preprocessing_steps=preprocessing_steps)


# Classifier possibilities and parameters
best_params = {
    'RandomForest': {
        'n_estimators': 200,
        'max_depth': 10,
        'class_weight': 'balanced'        
    },
    'XGB': {
        'n_estimators': 200,
        'max_depth': 2,
        'scale_pos_weight': 1/0.184,
        'objective': 'binary:logistic'       
    },
    'LGBM': {
        'n_estimators': 100,
        'num_leaves': 35,
        'reg_alpha': 10,
        'reg_lambda': 1,
        'max_depth': 7,
        'min_child_weight': 1e1,
        'min_child_samples': 20,
        'class_weight': 'balanced'   
    },
    'LogisticReg': {
        'C': 1.0,
        'class_weight': 'balanced'
    }
}
cv_params = {
    'RandomForest': {
        'max_depth': [1, 2]
    },
    'XGB': {
        'max_depth': [2, 3],
        'scale_pos_weight': [1/0.15, 1/0.184, 1/0.20]
    },
    'LGBM': {
        'reg_alpha': [1e-2, 1e0, 1e1, 1e2],
        'reg_lambda': [1e-2, 1e0, 1e1, 1e2],
    },
    'LogisticReg': {
        'C': [1e-1, 1e0, 1e1]
    }
}
est_list = {
    'RandomForest': RandomForestClassifier(**best_params['RandomForest']),
    'XGB': XGBClassifier(**best_params['XGB']),
    'LGBM': LGBMClassifier(**best_params['LGBM']),
    'LogisticReg': LogisticRegression(**best_params['LogisticReg'])
}

est_name = 'LGBM'


# Classification
clf = classify(
    x_tr=x_tr.values,
    y_tr=y_tr.values.ravel(),
    groups_tr=groups_tr.values,
    est=est_list[est_name],
    perform_cross_validation=perform_cross_validation,
    cv_params=cv_params[est_name],
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
