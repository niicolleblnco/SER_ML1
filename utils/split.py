import numpy as np

def stratified_splits(labels, test_size=0.2, val_size=0.1, seed=42):
    """
    Returns idx_train, idx_val, idx_test with class proportions preserved.

    labels: 1D array-like of ints, length N
    test_size: fraction of full set for test
    val_size: fraction of full set for validation
    """

    rng = np.random.default_rng(seed) # Initialize reproducible random number generator
    labels = np.asarray(labels, dtype=np.int64) # Ensure labels are a NumPy integer array
    n = labels.shape[0]

    # Validate split proportions
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0,1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1")

    # Identify unique classes
    classes = np.unique(labels)
    idx_train, idx_val, idx_test = [], [], []

    # Split indices separately for each class
    for c in classes:
        idx_c = np.where(labels == c)[0] # Indices belonging to class c
        rng.shuffle(idx_c)
        n_c = len(idx_c)

        # Number of samples per split for this class
        n_test = int(round(n_c * test_size))
        n_val = int(round(n_c * val_size))

        # Ensure bounds are respected
        n_test = min(n_test, n_c)
        n_val = min(n_val, n_c - n_test)

        # Assign splits
        test_c = idx_c[:n_test]
        val_c = idx_c[n_test:n_test + n_val]
        train_c = idx_c[n_test + n_val:]

        idx_test.append(test_c)
        idx_val.append(val_c)
        idx_train.append(train_c)

    # Concatenate indices from all classes
    idx_train = np.concatenate(idx_train) if len(idx_train) else np.array([], dtype=np.int64)
    idx_val = np.concatenate(idx_val) if len(idx_val) else np.array([], dtype=np.int64)
    idx_test = np.concatenate(idx_test) if len(idx_test) else np.array([], dtype=np.int64)

    # Shuffle each split independently
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()