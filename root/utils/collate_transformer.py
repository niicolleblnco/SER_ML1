import torch
import random

CHUNK_LEN = 300

def pad_and_mask(sequences, labels):
    lengths = [seq.shape[0] for seq in sequences]
    max_len = max(lengths)
    feat_dim = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, feat_dim, dtype=torch.float32)
    mask = torch.ones(len(sequences), max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        T_i = seq.shape[0]
        padded[i, :T_i] = seq
        mask[i, :T_i] = False

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels

def collate_transformer_train(batch):
    sequences = []
    labels = []

    for seq, lab in batch:
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)

        T = seq.shape[0]
        if T > CHUNK_LEN:
            start = random.randint(0, T - CHUNK_LEN)
            seq = seq[start:start + CHUNK_LEN]

        sequences.append(seq)
        labels.append(lab)

    return pad_and_mask(sequences, labels)

def collate_transformer_eval(batch):
    sequences = []
    labels = []

    for seq, lab in batch:
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)

        sequences.append(seq)
        labels.append(lab)

    return pad_and_mask(sequences, labels)