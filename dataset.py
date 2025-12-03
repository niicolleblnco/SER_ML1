import numpy as np
import torch
from torch.utils.data import Dataset

class PrecomputedMFCCDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.mfccs = data["mfccs"]
        self.labels = data["labels"]

        print("Loaded", len(self.mfccs), "MFCC sequences")

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.mfccs[idx]).float()
        y = int(self.labels[idx])
        return x, y