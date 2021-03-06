import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler


# Preprocessing wrappers
class TSFormatting(TransformerMixin):
    '''
    Reformat the data to feed the tsfresh package's feature computation.
    This includes computing the ISI (inter-spike interval) series and stacking
    the data vertically.
    '''
    def fit(self, X, y=None, **fit_params):
        return(self)

    def transform(self, X):
        data = X.drop(columns=['neuron_id'])
        # Stack vertically
        data = data.stack().reset_index()
        # Follow API naming convention
        data.columns=['id', 'time', 'val']
        # Convert 'timestamp_i' -> i
        data['time'] = data['time'].apply(lambda s: int(s[10:]))
        return(data)


class LowVarianceFeaturesRemover(BaseEstimator, TransformerMixin):
    '''
    Remove low-variance features based on thresholding
    '''
    def __init__(self, threshold=0.2, epsilon=1e-9):
        self.thr = threshold
        self.eps = epsilon
    
    def _reset(self):
        if hasattr(self, 'bool_to_keep_'):
            del self.bool_to_keep_
    
    def fit(self, X, y=None):
        self._reset()
        criterion = (np.std(X, axis=0) / (np.mean(X, axis=0) + self.eps))
        self.bool_to_keep_ = criterion > self.thr
        return(self)
    
    def transform(self, X):
        try:
            return(X[X.columns[self.bool_to_keep_]])
        except:
            return(X[:, self.bool_to_keep_])


class CenterScaler(BaseEstimator, TransformerMixin):
    '''
    Wrapper to have sklearn's StandardScaler return pandas DataFrame
    '''
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X.values)
        return(self)
    
    def transform(self, X):
        return(pd.DataFrame(
            data=self.scaler.transform(X.values),
            columns=X.columns,
            index=X.index
        ))


# Pre-processing 
def preprocess_data(x_tr, x_te, y_tr=None,
                    preprocessing_steps=None,
                    resampling_steps=None):
    '''
    Standard preprocessing pipeline: filter out missing columns, compute
    groups, apply various preprocessing steps.
    '''
    # Filter out incorrect columns
    missing_tr_columns = set(x_tr.columns[x_tr.isnull().any()])
    missing_te_columns = set(x_te.columns[x_te.isnull().any()])
    missing_columns = list(missing_tr_columns | missing_te_columns)

    x_tr.drop(missing_columns, axis=1, inplace=True)
    x_te.drop(missing_columns, axis=1, inplace=True)

    # Handle neuron_id as a group identifier
    groups_tr = x_tr['neuron_id']

    x_tr.drop(columns=['neuron_id'], inplace=True)
    x_te.drop(columns=['neuron_id'], inplace=True)

    # Center and scale if required
    if preprocessing_steps is not None:
        for prep_step in preprocessing_steps:
            prep_step.fit(x_tr)
            x_tr = prep_step.transform(x_tr)
            x_te = prep_step.transform(x_te)
    
    # Resample if required
    if resampling_steps is not None:
        for rsmp_step in resampling_steps:
            rsmp_step.set_params(return_indices=True)
            idx = rsmp_step.fit_resample(x_tr, y_tr)[2]
            x_tr = x_tr.iloc[idx]
            y_tr = y_tr.iloc[idx]
            groups_tr = groups_tr.iloc[idx]

    return(x_tr, x_te, groups_tr, y_tr)


def load_data(features_folder, data_folder='./data/'):
    x_tr = pd.read_csv(features_folder + 'feat_tr.csv', index_col=[0])
    x_te = pd.read_csv(features_folder + 'feat_te.csv', index_col=[0])
    y_tr = pd.read_csv(data_folder + 'target.csv', index_col=[0])
    return(x_tr, x_te, y_tr)