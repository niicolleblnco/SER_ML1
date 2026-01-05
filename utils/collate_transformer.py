import torch
import random

def pad_and_mask(sequences, labels):
    """
    sequences: list of (T_i, F) tensors
    labels:    list of ints

    Returns
      padded: (B, T_max, F)
      mask:   (B, T_max) bool, True means padding positions
      labels: (B,) long
    """
    if len(sequences) == 0:
        raise ValueError("Empty batch in collate")

    lengths = [int(seq.shape[0]) for seq in sequences]
    max_len = max(lengths)

    if max_len <= 0:
        raise ValueError("All sequences have zero length")

    feat_dim = int(sequences[0].shape[1])

    padded = torch.zeros((len(sequences), max_len, feat_dim), dtype=torch.float32)
    mask = torch.ones((len(sequences), max_len), dtype=torch.bool)

    for i, seq in enumerate(sequences):
        T_i = int(seq.shape[0])
        if T_i <= 0:
            continue
        padded[i, :T_i] = seq
        mask[i, :T_i] = False

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels


def collate_transformer_train(batch, chunk_len=300):
    """
    Randomly crops sequences longer than chunk_len.
    """
    sequences = []
    labels = []

    for seq, lab in batch:
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)

        T = int(seq.shape[0])
        if chunk_len is not None and T > chunk_len:
            start = random.randint(0, T - chunk_len)
            seq = seq[start:start + chunk_len]

        sequences.append(seq)
        labels.append(int(lab))

    return pad_and_mask(sequences, labels)


def collate_transformer_eval(batch):
    """
    Keeps full sequences.
    """
    sequences = []
    labels = []

    for seq, lab in batch:
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)

        sequences.append(seq)
        labels.append(int(lab))

    return pad_and_mask(sequences, labels)