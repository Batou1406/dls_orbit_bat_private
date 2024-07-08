import torch
from torch.utils.data import Dataset, DataLoader
import glob

class ObservationActionDataset(Dataset):
    """Cutsom dataloader to load generated dataset
    Args :
        file_path : The file path to the dataset"""
    def __init__(self, file_path):
        data = torch.load(file_path)
        self.observations = data['observations']
        self.actions = data['actions']

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

class ChunkedObservationActionDataset(Dataset):
    """Cutsom dataloader to load generated chunked dataset
    Args :
        file_path : The file path to the dataset as regex expression with the chunk"""
    def __init__(self, file_pattern):
        self.files = glob.glob(file_pattern)
        self.data = []

        for file in self.files:
            self.data.append(torch.load(file))

        self.observations = torch.cat([d['observations'] for d in self.data])
        self.actions = torch.cat([d['actions'] for d in self.data])

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]