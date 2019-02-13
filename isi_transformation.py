import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


class ISIFormatting(BaseEstimator, TransformerMixin):
    '''
    Reformat the array of spike times to the ISI profiles
    '''
    def fit(self, X, y=None, **fit_params):
        return(self)
    
    def transform(self, X):
        data = X.copy()
        # Create a 0 timestamp to correctly compute the first "interval" value
        data['initial_zero'] = 0.
        # Compute the ISI profile
        data = data[['initial_zero'] + X.columns.tolist()[1:]]
        print(data.head())
        data = data.diff(axis=1)
        print(data.head())
        # Drop the NaN column
        data.drop(columns=['initial_zero'], inplace=True)
        # Keep the neuron_id column
        data['neuron_id'] = X['neuron_id']
        return(data)


# Parameters
data_folder = './data/'
features_folder = './features/isi/'


# Load data
x_tr = pd.read_csv(data_folder + 'training.csv', index_col=[0])
x_te = pd.read_csv(data_folder + 'input_test.csv', index_col=[0])


# Transform to ISI representation
feat_tr = ISIFormatting().transform(x_tr)
feat_te = ISIFormatting().transform(x_te)


# Save the computed features
feat_tr.to_csv(features_folder + 'feat_tr.csv', header=True, index=True)
feat_te.to_csv(features_folder + 'feat_te.csv', header=True, index=True)