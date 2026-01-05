import argparse
import pandas as pd
from pathlib import Path
import re

# Regular expression that matches the official RAVDESS filename format.
# Example filename:
# 03-01-05-02-02-01-12.wav
RAVDESS_RE = re.compile(
    r"(?P<modality>\d{2})-"        # modality code
    r"(?P<vocal_ch>\d{2})-"        # vocal channel
    r"(?P<emotion>\d{2})-"         # emotion label
    r"(?P<em_int>\d{2})-"          # emotional intensity
    r"(?P<statement>\d{2})-"       # statement identifier
    r"(?P<rep>\d{2})-"             # repetition
    r"(?P<actor>\d{2})\.(wav|WAV)$" # actor ID and file extension
)

# Mapping from RAVDESS emotion codes to integer class labels.
# These indices are typically used for classification targets.
RAVDESS_EMO_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7   # surprised
}

def build_df_from_ravdess(root: str) -> pd.DataFrame:
    """
    Scans a directory recursively for RAVDESS .wav files and
    returns a DataFrame with absolute file paths and emotion labels.

    Parameters
    ----------
    root : str
        Root directory containing the RAVDESS dataset.

    Returns
    -------
    pd.DataFrame
        Columns:
          - Path: absolute path to the audio file
          - Emotion: integer emotion label
    """
    root = Path(root)
    rows = []

    # Recursively search for .wav files
    for p in root.rglob("*.wav"):
        # Match filename against the RAVDESS naming convention
        m = RAVDESS_RE.match(p.name)
        if not m:
            continue

        # Extract emotion code from filename
        emo_code = m.group("emotion")
        if emo_code not in RAVDESS_EMO_MAP:
            continue

        # Store absolute path and mapped emotion label
        rows.append({
            "Path": str(p.resolve()),
            "Emotion": RAVDESS_EMO_MAP[emo_code]
        })

    # Fail early if no valid files were found
    if not rows:
        raise RuntimeError(f"No RAVDESS-style .wav files found under {root}")

    # Create DataFrame, sort for reproducibility, reset index
    df = pd.DataFrame(rows).sort_values("Path").reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Simple CLI entry point to validate dataset structure
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    # Build the DataFrame. This will raise an error if the dataset is invalid.
    _ = build_df_from_ravdess(args.data_root)