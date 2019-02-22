import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

from sklearn.neighbors import KNeighborsClassifier
from distances import kolmogorov_smirnov, kolmogorov_smirnov_opt

from imblearn.under_sampling import RandomUnderSampler
from preprocessing import load_data, preprocess_data
from classification import classify


# Parameters
data_folder = './data/'
isi_folder = './features/isi/'
cdf_folder = './features/CDF/'
submission_folder = './submissions/neighbors_isi/'

recompute_features = False
use_precomputed = False

perform_evaluation = False
perform_cross_validation = True

compute_submission = False


###############################################################################
#                                                                             #
#                        Features pre-computation part                        #
#                                                                             #
###############################################################################

features_to_compute = None

if recompute_features:
    if features_to_compute == 'KS':
        x_tr, x_te, y_tr = load_data(
            features_folder=isi_folder,
            data_folder=data_folder
        )

        x_tr_val = np.sort(x_tr.drop(columns=['neuron_id']).values, axis=1)
        x_te_val = np.sort(x_te.drop(columns=['neuron_id']).values, axis=1)

        feat_tr_val = pairwise_distances(
            X=x_tr_val,
            metric=kolmogorov_smirnov_opt,
            n_jobs=4
        )

        feat_tr = pd.DataFrame(data=feat_tr_val, index=x_tr.index)
        feat_tr['neuron_id'] = x_tr['neuron_id']
        feat_tr.to_csv('./features/KS/feat_tr.csv', index=True, header=True)
        del feat_tr


        feat_te_val = cdist(x_te_val, x_tr_val, metric=kolmogorov_smirnov_opt)

        feat_te = pd.DataFrame(data=feat_te_val, index=x_te.index)
        feat_te['neuron_id'] = x_te['neuron_id']
        feat_te.to_csv('./features/KS/feat_te.csv', index=True, header=True)
        del feat_te
    
    elif features_to_compute == 'CDF':
        x_tr, x_te, y_tr = load_data(
            features_folder=isi_folder,
            data_folder=data_folder
        )

        x_tr_val = x_tr.drop(columns=['neuron_id']).values
        x_te_val = x_te.drop(columns=['neuron_id']).values

        _, full_bins = np.histogram(np.log(x_tr_val.ravel()), bins=100)

        f = lambda a: np.histogram(a, bins=full_bins, density=True)[0]
        print('Computing training features...')
        feat_tr_val = np.cumsum(
            np.apply_along_axis(func1d=f, axis=1, arr=x_tr_val),
            axis=1
        )
        print('Computing test features...')
        feat_te_val = np.cumsum(
            np.apply_along_axis(func1d=f, axis=1, arr=x_te_val),
            axis=1
        )

        feat_tr = pd.DataFrame(data=feat_tr_val, index=x_tr.index)
        feat_tr['neuron_id'] = x_tr['neuron_id']
        feat_tr.to_csv('./features/' + features_to_compute + '/feat_tr.csv', index=True, header=True)
        del feat_tr

        feat_te = pd.DataFrame(data=feat_te_val, index=x_te.index)
        feat_te['neuron_id'] = x_te['neuron_id']
        feat_te.to_csv('./features/' + features_to_compute + '/feat_te.csv', index=True, header=True)
        del feat_te

###############################################################################
#                                                                             #
#                             Classification part                             #
#                                                                             #
###############################################################################

# Classifier possibilities and parameters
best_params = {
    'Euclidean': {
        'n_neighbors': 10,
        'n_jobs': -1,
    },
    'MinkowskiL1': {
        'n_neighbors': 10,
        'n_jobs': -1,
    },
    'KS': {
        'n_neighbors': 10,
        'n_jobs': -1,
    },
    'CDF_L2': {
        'n_neighbors': 50,
        'weights': 'uniform',
        'leaf_size': 15,
        'n_jobs': -1
    }
}
cv_params = {
    'Euclidean': {
        'n_neighbors': [3, 5, 7]
    },
    'MinkowskiL1': {
        'n_neighbors': [5, 10, 15]
    },
    'KS': {
        'n_neighbors': [5, 10, 15]
    },
    'CDF_L2': {
        'n_neighbors': [50],
        'weights': ['uniform'],
        'leaf_size': [15],
        'p': [2]
    }
}
est_list = {
    'Euclidean': KNeighborsClassifier(**best_params['Euclidean'], p=2),
    'MinkowskiL1': KNeighborsClassifier(**best_params['MinkowskiL1'], p=1),
    'KS': KNeighborsClassifier(**best_params['KS'], metric=kolmogorov_smirnov),
    'CDF_L2': KNeighborsClassifier(**best_params['CDF_L2'], p=2),
}

est_name = 'CDF_L2'


# Make sure we're not trying to load unavailable pre-computed features
if est_name not in ['KS']:
    use_precomputed = False


# Load features
if use_precomputed:
    x_tr, x_te, y_tr = load_data(
        features_folder='./features/' + est_name + '/',
        data_folder=data_folder
    )
    est_list[est_name].set_params(metric='precomputed')
elif est_name[:3] == 'CDF':
    x_tr, x_te, y_tr = load_data(
        features_folder=cdf_folder,
        data_folder=data_folder
    )
else:
    x_tr, x_te, y_tr = load_data(
        features_folder=isi_folder,
        data_folder=data_folder
    )


# Pre-process
preprocessing_steps = []
resampling_steps = [RandomUnderSampler()]
x_tr, x_te, groups_tr, y_tr = preprocess_data(
    x_tr,
    x_te,
    y_tr=y_tr,
    preprocessing_steps=preprocessing_steps,
    resampling_steps=resampling_steps
)


# Pre-sort the values to speed-up distance computation
if not use_precomputed:
    if est_name in ['KS']:
        x_tr.iloc[:, :] = np.sort(x_tr.values, axis=1)
        x_te.iloc[:, :] = np.sort(x_te.values, axis=1)


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
    random_state=None
)