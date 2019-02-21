import numpy as np
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, norm, is_train):
        self.data = data
        self.target = target
        self.norm = norm
        self.is_train = is_train

    def __len__(self):
        return(self.data.shape[0])

    def __getitem__(self, idx):
        data_idx = np.expand_dims(self.data[idx, :], axis=0)
        target_idx = self.target[idx]
        return(data_idx, target_idx)
