import torch
import random

CHUNK_LEN = 300

def collate_transfomer(batch):
    sequences = []
    labels = []

    for seq, lab in batch:
        if not isinstance(seq, torch.Tensor):
            seq = torch.Tensor(seq, dtype=torch.float32)
        
        T = seq.shape[0]

        if T > CHUNK_LEN:
            start = random.randint(0, T - CHUNK_LEN)
            seq = seq[start:start + CHUNK_LEN]
        
        sequences.append(seq)
        labels.append(lab)
    
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths)
    n_mfcc = sequences[0].shape[1]

    padded = torch.zeros(len(batch), max_len, n_mfcc, dtype=torch.float32)
    mask = torch.ones(len(batch), max_len, dtype=bool)

    for i, seq in enumerate(sequences):
        T_i = seq.shape[0]
        padded[i, :T_i] = seq
        mask[i, :T_i] = False

    labels = torch.tensor(labels, dtype=torch.long)

    return padded, mask, labels

def collate_transformer_eval(batch):
    sequences = []
    labels = []

    for seq, lab in batch:
        if not isinstance(seq, torch.Tensor):
            seq = torch.Tensor(seq, dtype=torch.float32)
        sequences.append(seq)
        labels.append(lab)

    
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths)
    n_mfcc = sequences[0].shape[1]
    
    padded = torch.zeros(len(batch), max_len, n_mfcc, dtype=torch.float32)
    mask = torch.ones(len(batch), max_len, dtype=bool)

    for i, seq in enumerate(sequences):
        T_i = seq.shape[0]
        padded[i, :T_i] = seq
        mask[i, :T_i] = False

    labels = torch.tensor(labels, dtype=torch.long)

    return padded, mask, labels