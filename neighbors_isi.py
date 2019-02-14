import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from distances import kolmogorov_smirnov

from imblearn.under_sampling import RandomUnderSampler
from preprocessing import preprocess_data
from classification import classify


# Parameters
data_folder = './data/'
isi_folder = './features/isi/'
submission_folder = './submissions/neighbors_isi/'

perform_cross_validation = False

compute_submission = False


###############################################################################
#                                                                             #
#                             Classification part                             #
#                                                                             #
###############################################################################

# Load features
x_tr = pd.read_csv(isi_folder + 'feat_tr.csv', index_col=[0])
x_te = pd.read_csv(isi_folder + 'feat_te.csv', index_col=[0])

y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])


# Pre-processing
preprocessing_steps = []
resampling_steps = [RandomUnderSampler()]
x_tr, x_te, groups_tr, y_tr = preprocess_data(
    x_tr,
    x_te,
    y_tr=y_tr,
    preprocessing_steps=preprocessing_steps,
    resampling_steps=resampling_steps
)
print(len(x_tr), y_tr.mean())


# Classifier possibilities and parameters
best_params = {
    'EuclideanKNN': {
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
    }
}
cv_params = {
    'EuclideanKNN': {
        'n_neighbors': [3, 5, 7]
    },
    'MinkowskiL1': {
        'n_neighbors': [5, 10, 15]
    },
    'KS': {
        'n_neighbors': [5, 10, 15]
    }
}
est_list = {
    'EuclideanKNN': KNeighborsClassifier(**best_params['EuclideanKNN'], p=2),
    'MinkowskiL1': KNeighborsClassifier(**best_params['MinkowskiL1'], p=1),
    'KS': KNeighborsClassifier(**best_params['KS'], metric=kolmogorov_smirnov),
}

est_name = 'MinkowskiL1'


# Classification
if est_name == 'KS':
    clf = classify(
        x_tr=np.sort(x_tr.values, axis=1),
        y_tr=y_tr.values.ravel(),
        groups_tr=groups_tr.values,
        est=est_list[est_name],
        perform_cross_validation=perform_cross_validation,
        cv_params=cv_params[est_name],
        random_state=42
    )
else:
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


# Compute submission
if compute_submission:
    clf.fit(x_tr.values, y_tr.values.ravel())
    y_te_pred = clf.predict(x_te)
    y_te_pred_df = pd.DataFrame(data=y_te_pred, columns=['TARGET'], index=(x_te.index))
    y_te_pred_df.index.name = 'ID'
    y_te_pred_df.to_csv(submission_folder + 'y_te_pred.csv', header=True, index=True)
