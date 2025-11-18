import numpy as np
from sklearn.model_selection import train_test_split

def stratified_splits(df, label_col="Emotion", test_size=0.2, val_size=0.1, seed=42):
    idx = np.arange(len(df))
    y = df[label_col].values

    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )

    val_rel = val_size / (1.0 - test_size)
    y_train_val = y[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_rel, stratify=y_train_val, random_state=seed
    )

    return train_idx, val_idx, test_idx