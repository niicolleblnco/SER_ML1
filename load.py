import argparse
import pandas as pd
from pathlib import Path

CREMA_EMO_MAP = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

def build_df_from_crema(root):
    root = Path(root)
    rows = []
    for p in root.rglob("*.wav"):
        name = p.stem
        parts = name.split("_")

        if len(parts) < 3:
            continue

        emo = parts[2].upper()
        if emo not in CREMA_EMO_MAP:
            continue

        rows.append({
            "Path": str(p.resolve()),
            "Emotion": CREMA_EMO_MAP[emo]
        })

    df = pd.DataFrame(rows).sort_values("Path").reset_index(drop=True)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    df = build_df_from_crema(args.data_root)