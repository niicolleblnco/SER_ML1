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

def build_df(data_root):
    rows = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.lower().endswith(".wav"):
                rows.append(os.path.join(root, f))
    if not rows:
        raise RuntimeError("No wav files found")
    df = pd.DataFrame({"Path": sorted(rows)})
    df["Emotion"] = df["Path"].apply(lambda p: int(os.path.basename(p)[6:8]) - 1)
    return df

def extract_all(df):
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

    feats = []
    labels = []

    for path, label in tqdm(zip(df["Path"], df["Emotion"]), total=len(df)):
        wav, sr = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            wav = resampler(wav)

        max_abs = wav.abs().max()
        if max_abs > 0:
            wav = wav / max_abs

        mfcc = mfcc_tf(wav).squeeze(0)
        global_mean = mfcc.mean(dim=1)
        feats.append(global_mean.numpy())
        labels.append(label)

    return np.vstack(feats), np.array(labels)

def main():
    df = build_df(DATA_ROOT)

    feats, labels = extract_all(df)

    np.save("features.npy", feats)
    np.save("labels.npy", labels)
    print("Saved features.npy and labels.npy")
    print("Feature shape", feats.shape)

if __name__ == "__main__":
    main()