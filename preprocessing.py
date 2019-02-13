import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler


class TSFormatting(TransformerMixin):
    '''
    Reformat the data to feed the tsfresh package's feature computation.
    This includes computing the ISI (inter-spike interval) series and stacking
    the data vertically.
    '''
    def fit(self, X, y=None, **fit_params):
        return(self)

    def transform(self, X):
        data = X.copy()
        # Create a 0 timestamp to correctly compute the first "interval" value
        data['initial_zero'] = 0
        # Stack vertically with expected order of columns
        data = data[['initial_zero'] + X.columns.tolist()[1:]].stack().reset_index()
        # Follow API naming convention
        data.columns=['id', 'time', 'val']
        # Compute the interval values
        data['val'] = data.groupby('id')['val'].diff()
        # Drop the first value
        data.dropna(inplace=True)
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


def preprocess_data(x_tr, x_te, preprocessing_steps=None):
    '''
    Standard preprocessing pipeline: filter out missing columns, compute
    groups, apply various preprocessing steps.
    '''
    # Filter out incorrect columns
    X_tr = x_tr.copy()
    X_te = x_te.copy()

    missing_tr_columns = set(X_tr.columns[X_tr.isnull().any()])
    missing_te_columns = set(X_te.columns[X_te.isnull().any()])
    missing_columns = list(missing_tr_columns | missing_te_columns)

    X_tr.drop(missing_columns, axis=1, inplace=True)
    X_te.drop(missing_columns, axis=1, inplace=True)

    # Handle neuron_id as a group identifier
    groups_tr = X_tr['neuron_id']

    X_tr.drop(columns=['neuron_id'], inplace=True)
    X_te.drop(columns=['neuron_id'], inplace=True)

    # Center and scale if required
    if preprocessing_steps is not None:
        for prep_step in preprocessing_steps:
            prep_step.fit(X_tr)
            X_tr = prep_step.transform(X_tr)
            X_te = prep_step.transform(X_te)

    return(X_tr, X_te, groups_tr)