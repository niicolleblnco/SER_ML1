import argparse
import os
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from tqdm import tqdm
from load import build_df_from_ravdess


def build_mfcc_transform(target_sr, win_ms, hop_ms, n_mels, n_mfcc):
    """
    Builds a torchaudio MFCC transform with fixed parameters.

    This transform is reused for all files to avoid
    re-instantiating it for every audio sample.
    """
    # Convert window and hop size from milliseconds to samples
    win_length = int(target_sr * win_ms / 1000)
    hop_length = int(target_sr * hop_ms / 1000)

    # Use the smallest power-of-two FFT size >= window length
    n_fft = 1 << (win_length - 1).bit_length()

    return T.MFCC(
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


def extract_mfcc(waveform, mfcc_tf):
    """
    Extracts MFCC features from a mono waveform.

    Parameters
    ----------
    waveform : torch.Tensor
        Shape (1, N), mono audio signal.
    mfcc_tf : torchaudio.transforms.MFCC
        Pre-built MFCC transform.

    Returns
    -------
    np.ndarray
        Shape (T, n_mfcc), where T is the number of frames.
    """
    # Peak normalization to avoid scale differences between files
    max_abs = waveform.abs().amax()
    if max_abs > 0:
        waveform = waveform / max_abs

    # Apply MFCC transform
    mfcc = mfcc_tf(waveform).squeeze(0)       # (n_mfcc, T)

    # Transpose to time-major format expected by the model
    return mfcc.transpose(0, 1).cpu().numpy() # (T, n_mfcc)


def main():
    # Command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/features/ravdess/precomputed_mfcc.npz")
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--win-ms", type=int, default=25)
    ap.add_argument("--hop-ms", type=int, default=10)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--n-mfcc", type=int, default=20)
    args = ap.parse_args()

    # Ensure output directory exists
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # Build dataframe with file paths and emotion labels
    df = build_df_from_ravdess(args.data_root)
    print("Extracting MFCCs from", len(df), "files")

    # Create MFCC transform once
    mfcc_tf = build_mfcc_transform(
        args.target_sr,
        args.win_ms,
        args.hop_ms,
        args.n_mels,
        args.n_mfcc
    )

    # Cache resamplers to avoid recreating them repeatedly
    resampler_cache = {}

    mfcc_list = []
    labels = []
    lengths = []

    # Iterate over all audio files
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MFCC extraction"):
        path = row["Path"]
        label = int(row["Emotion"])

        # Load audio file
        waveform_np, orig_sr = sf.read(path, always_2d=True)
        waveform = torch.from_numpy(waveform_np.T).float()

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if sampling rate differs
        if orig_sr != args.target_sr:
            key = (orig_sr, args.target_sr)
            if key not in resampler_cache:
                resampler_cache[key] = T.Resample(orig_sr, args.target_sr)
            waveform = resampler_cache[key](waveform)

        # Extract MFCC features
        mfcc = extract_mfcc(waveform, mfcc_tf)

        # Skip empty outputs
        if mfcc.shape[0] <= 0:
            continue

        mfcc_list.append(mfcc)
        labels.append(label)
        lengths.append(mfcc.shape[0])

    # Fail early if no features were extracted
    if len(mfcc_list) == 0:
        raise RuntimeError("No MFCCs extracted, check your dataset path and audio files")

    # Store variable-length MFCCs as an object array
    mfcc_obj = np.array(mfcc_list, dtype=object)

    # Save features, labels, and sequence lengths
    np.savez(
        args.out,
        mfccs=mfcc_obj,
        labels=np.array(labels, dtype=np.int64),
        lengths=np.array(lengths, dtype=np.int64)
    )

    print("Saved", len(mfcc_list), "items to", args.out)


if __name__ == "__main__":
    main()