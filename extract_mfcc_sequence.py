import numpy as np
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from loady import build_df_from_ravdess

DATA_ROOT = "crema"
SAMPLE_RATE = 16000
WIN_MS = 25
HOP_MS = 10
N_MELS = 64
N_MFCC = 20

def extract_mfcc(waveform, sr):
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # normalize
    max_abs = waveform.abs().amax()
    if max_abs > 0:
        waveform = waveform / max_abs

    # MFCC settings
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
            center=True,
            pad_mode="reflect",
            power=2.0,
        ),
    )

    mfcc = mfcc_tf(waveform).squeeze(0)      # (n_mfcc, T)
    mfcc = mfcc.transpose(0, 1)              # (T, n_mfcc)
    return mfcc.cpu().numpy()

def main():
    print("Loading file list...")
    df = build_df_from_ravdess(DATA_ROOT)

    mfcc_list = []
    labels = []

    print("Extracting MFCC sequences...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        path = row["Path"]
        label = int(row["Emotion"])

        waveform, sr = torchaudio.load(path)
        mfcc = extract_mfcc(waveform, sr)

        mfcc_list.append(mfcc)
        labels.append(label)

    print("Saving precomputed_mfcc.npz...")
    np.savez(
        "precomputed_mfcc.npz",
        mfccs=np.array(mfcc_list, dtype=object),
        labels=np.array(labels, dtype=np.int64)
    )

    print("Done. Saved precomputed_mfcc.npz")

if __name__ == "__main__":
    main()