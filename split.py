import numpy as np
from sklearn.model_selection import train_test_split

def stratified_splits(labels, label_col=None, test_size=0.2, val_size=0.1, seed=42):
    y = labels 

    idx = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    train_y = y[train_idx]

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size / (1 - test_size),
        stratify=train_y,
        random_state=seed
    )

    return train_idx, val_idx, test_idx