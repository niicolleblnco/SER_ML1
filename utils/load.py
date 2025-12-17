import argparse
import pandas as pd
from pathlib import Path
import re

RAVDESS_RE = re.compile(r"(?P<modality>\d{2})-(?P<vocal_ch>\d{2})-(?P<emotion>\d{2})-"
                        r"(?P<em_int>\d{2})-(?P<statement>\d{2})-(?P<rep>\d{2})-"
                        r"(?P<actor>\d{2})\.(wav|WAV)$")

RAVDESS_EMO_MAP = {
    "01": 0, # neutral
    "02": 1, # calm
    "03": 2, # happy
    "04": 3, # sad
    "05": 4, # angry
    "06": 5, # fearful
    "07": 6, # disgust
    "08": 7  # surprised
}
def build_df_from_ravdess(root: str) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in root.rglob("*.wav"):
        m = RAVDESS_RE.match(p.name)
        if not m:
            continue
        emo_code = m.group("emotion")
        if emo_code not in RAVDESS_EMO_MAP:
            continue
        rows.append({"Path": str(p.resolve()), "Emotion": RAVDESS_EMO_MAP[emo_code]})
    if not rows:
        raise RuntimeError(f"No RAVDNESS-style .wav files found under {root}")
    df = pd.DataFrame(rows).sort_values("Path").reset_index(drop=True)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    df = build_df_from_ravdess(args.data_root)