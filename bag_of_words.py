import numpy as np
import pandas as pd

from pyts.classification import BOSSVSClassifier, SAXVSMClassifier

from preprocessing import load_data, LowVarianceFeaturesRemover, preprocess_data
from classification import classify

import sys


# Parameters
data_folder = './data/'
isi_folder = './features/isi/'
submission_folder = './submissions/bag_of_words/'

perform_evaluation = True
perform_cross_validation = False

compute_submission = False


###############################################################################
#                                                                             #
#                             Classification part                             #
#                                                                             #
###############################################################################

# Classifier possibilities and parameters
best_params = {
    'BOSSVS': {
        'quantiles': 'empirical',       # Instead of N(0,1) quantiles
        'norm_mean': False,             # Don't center
        'norm_std': False,              # Don't scale
        'smooth_idf': True,             # Prevent division by zero, but bias
        'sublinear_tf': False,          # Disable sublinear tf scaling
        'n_coefs': None,                # Mandatory number of Fourier coefs
        'window_size': 3,               # Mandatory window size for features
    },
    'SAXSVM': {
        'quantiles': 'empirical',
        'numerosity_reduction': False,
        'use_idf': True,
        'n_bins': 4,
        'window_size': 4
    }
}
cv_params = {
    'BOSSVS': {
        'n_coefs': [None]
    },
    'SAXSVM': {
        'n_bins': [4]
    }
}
est_list = {
    'BOSSVS': BOSSVSClassifier(**best_params['BOSSVS']),
    'SAXSVM': SAXVSMClassifier(**best_params['SAXSVM'])
}

est_name = 'BOSSVS'


# Load features
x_tr, x_te, y_tr = load_data(
    features_folder=isi_folder,
    data_folder=data_folder
)


# Pre-process
preprocessing_steps = []
resampling_steps = []
x_tr, x_te, groups_tr, y_tr = preprocess_data(
    x_tr,
    x_te,
    y_tr=y_tr,
    preprocessing_steps=preprocessing_steps,
    resampling_steps=resampling_steps
)


# Classification
clf = classify(
    est=est_list[est_name],
    x_tr=x_tr.values,
    y_tr=y_tr.values.ravel(),
    groups_tr=groups_tr.values,
    x_te=x_te.values,
    test_index=x_te.index,
    perform_evaluation=perform_evaluation,
    perform_cross_validation=perform_cross_validation,
    cv_params=cv_params[est_name],
    compute_submission=compute_submission,
    submission_path=(submission_folder + 'y_te_pred.csv'),
    random_state=42
)