import numpy as np
import torch
from torch.utils.data import Dataset

class PrecomputedMFCCDataset(Dataset):
    """
    Expects NPZ keys:
      mfccs: object array of length N, each item is (T_i, F) float
      labels: (N,) int
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.mfccs = data["mfccs"]
        self.labels = data["labels"]
        print("Loaded", len(self.mfccs), "MFCC sequences")

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        x_np = self.mfccs[idx]
        if x_np is None:
            raise ValueError(f"MFCC at index {idx} is None")

        x = torch.from_numpy(x_np).float()
        y = int(self.labels[idx])
        return x, y