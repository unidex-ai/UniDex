from torch.utils.data import Dataset
from src.utils.normalizers import Normalizer
import numpy as np
from tqdm import tqdm
import copy

class MixtureDataset(Dataset):
    def __init__(self,  **kwargs):
        self.datasets = list(kwargs.values())
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.lengths)
        self.normalizer = Normalizer(normalizers=[ds.normalizer for ds in self.datasets if hasattr(ds, 'normalizer')]) if not any(ds.normalizer is None for ds in self.datasets) else None

        for ds in self.datasets:
            ds.normalizer = self.normalizer

        self.shape = {}
        for ds in self.datasets:
            ds_shape = ds.shape
            for key, value in ds_shape.items():
                if key not in self.shape:
                    self.shape[key] = value
                else:
                    self.shape[key] = max(self.shape[key], value)

        self.state_dim = self.shape['state'][-1]
        self.action_dim = self.shape['action'][-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                data = self.datasets[i][idx]
                for key in data:
                    if key in self.shape and data[key].shape != self.shape[key]:
                        pad_width = [(0, self.shape[key][j] - data[key].shape[j]) for j in range(len(data[key].shape))]
                        data[key] = np.pad(data[key], pad_width, mode='constant', constant_values=0)
                return data
                        
            idx -= length
        raise IndexError("Index out of range")