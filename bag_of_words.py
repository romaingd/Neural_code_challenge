import numpy as np
import pandas as pd

from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pyts.classification import BOSSVSClassifier

from sklearn.metrics import cohen_kappa_score, make_scorer

from preprocessing import LowVarianceFeaturesRemover, preprocess_data
from classification import classify
from results_exploration import plot_avg_feature_importance

import sys


# Parameters
data_folder = './data/'
submission_folder = './submissions/benchmark/'

perform_cross_validation = False

compute_submission = False


###############################################################################
#                                                                             #
#                             Classification part                             #
#                                                                             #
###############################################################################

# Load features
x_tr = pd.read_csv(data_folder + 'training.csv', index_col=[0])
x_te = pd.read_csv(data_folder + 'input_test.csv', index_col=[0])

y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])

n_tr = len(x_tr)
n_te = len(x_te)
print(n_tr, 'training samples /', n_te, 'test samples\n')


# Pre-processing
preprocessing_steps = []
x_tr, x_te, groups_tr = preprocess_data(x_tr, x_te, preprocessing_steps=preprocessing_steps)


# Classifier possibilities and parameters
est_list = [
    BOSSVSClassifier(n_coefs=None, window_size=3)
]
cv_params = [
    {   # BOSSVSClassifier
        'n_coefs': [None]
    }
]
best_params = [
    {   # BOSSVSClassifier
        'quantiles': 'empirical',       # Instead of N(0,1) quantiles
        'norm_mean': False,             # Don't center
        'norm_std': False,              # Don't scale
        'smooth_idf': True,             # Prevent division by zero, but bias
        'sublinear_tf': False           # Disable sublinear tf scaling
    }
]
est_idx = 0


# Classification
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


# Compute submission
if compute_submission:
    clf.fit(x_tr.values, y_tr.values.ravel())
    y_te_pred = clf.predict(x_te)
    y_te_pred_df = pd.DataFrame(data=y_te_pred, columns=['TARGET'], index=(x_te.index))
    y_te_pred_df.index.name = 'ID'
    y_te_pred_df.to_csv(submission_folder + 'y_te_pred.csv', header=True, index=True)
