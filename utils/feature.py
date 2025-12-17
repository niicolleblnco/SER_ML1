import argparse
import os
import numpy as np
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from load import build_df_from_ravdess

def extract_mfcc(waveform, target_sr, win_ms, hop_ms, n_mels, n_mfcc):
    max_abs = waveform.abs().amax()
    if max_abs > 0:
        waveform = waveform / max_abs

    win_length = int(target_sr * win_ms / 1000)
    hop_length = int(target_sr * hop_ms / 1000)
    n_fft = 1 << (win_length - 1).bit_length()

    mfcc_tf = T.MFCC(
        sample_rate=target_sr,
        n_mfcc=n_mfcc,
        melkwargs=dict(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0
        )
    )

    mfcc = mfcc_tf(waveform).squeeze(0)              # (n_mfcc, T)
    return mfcc.transpose(0, 1).cpu().numpy()        # (T, n_mfcc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/features/ravdess/precomputed_mfcc.npz")
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--win-ms", type=int, default=25)
    ap.add_argument("--hop-ms", type=int, default=10)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--n-mfcc", type=int, default=20)
    args = ap.parse_args()

    outdir = os.path.dirname(args.out)
    if outdir != "":
        os.makedirs(outdir, exist_ok=True)

    df = build_df_from_ravdess(args.data_root)
    print(f"Extracting MFCCs from {len(df)} files...")

    mfcc_list = []
    labels = []

    # Extract MFCCs first, don't pad yet
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MFCC extraction"):
        path = row["Path"]
        label = int(row["Emotion"])

        waveform, orig_sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if orig_sr != args.target_sr:
            waveform = T.Resample(orig_sr, args.target_sr)(waveform)

        mfcc = extract_mfcc(
            waveform,
            args.target_sr,
            args.win_ms,
            args.hop_ms,
            args.n_mels,
            args.n_mfcc
        )

        mfcc_list.append(mfcc)
        labels.append(label)

    # Compute max length AFTER collecting MFCCs
    max_len = max(m.shape[0] for m in mfcc_list)
    feat_dim = mfcc_list[0].shape[1]

    # Create padded array
    padded = np.zeros((len(mfcc_list), max_len, feat_dim), dtype=np.float32)

    for i, m in enumerate(mfcc_list):
        padded[i, :m.shape[0]] = m

    print(f"Saving padded MFCCs to {args.out}")

    np.savez(
        args.out,
        mfccs=padded,
        labels=np.array(labels, dtype=np.int64)
    )


if __name__ == "__main__":
    main()