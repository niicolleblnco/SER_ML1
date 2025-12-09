import os
import numpy as np
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm

DATA_ROOT = "data"
SAMPLE_RATE = 16000
N_MFCC = 20
N_MELS = 64
WIN_MS = 25
HOP_MS = 10


def extract_mfcc_sequence(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    max_abs = waveform.abs().max()
    if max_abs > 0:
        waveform = waveform / max_abs

    win_length = int(SAMPLE_RATE * WIN_MS / 1000)
    hop_length = int(SAMPLE_RATE * HOP_MS / 1000)
    n_fft = 1 << (win_length - 1).bit_length()

    mfcc_tf = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs=dict(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=N_MELS,
            f_min=0,
            f_max=None,
            center=True,
            pad_mode="reflect",
            power=2.0,
        ),
    )

    mfcc = mfcc_tf(waveform).squeeze(0)   # shape (n_mfcc, T)
    mfcc = mfcc.transpose(0, 1)           # shape (T, n_mfcc)
    return mfcc

def main():
    mfcc_list = []
    labels = []
    lengths = []

    for root, _, files in os.walk(DATA_ROOT):
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                path = os.path.join(root, f)

                label = int(f.split("-")[2]) - 1
                labels.append(label)

                mfcc = extract_mfcc_sequence(path)
                mfcc_list.append(mfcc)
                lengths.append(mfcc.shape[0])

    np.savez(
        "precomputed_mfcc.npz",
        mfccs=np.array(mfcc_list, dtype=object),
        labels=np.array(labels, dtype=np.int64),
        lengths=np.array(lengths, dtype=np.int32),
    )

    print("Saved precomputed_mfcc.npz")
    print("Example MFCC shape:", mfcc_list[0].shape)

if __name__ == "__main__":
    main()