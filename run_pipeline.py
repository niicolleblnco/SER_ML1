import argparse
import subprocess
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="runs/exp1")
    args = ap.parse_args()

    # 1. Extract MFCC
    print("Extracting MFCCs if needed...")
    MFCC_PATH = "data/features/precomputed_mfcc.npz"

    if not os.path.exists(MFCC_PATH):
        subprocess.run([
            "python",
            "feature.py",
            "--data-root", args.data_root,
            "--out", "data/features/precomputed_mfcc.npz"
        ], check=True)
 
    print("MFCC extraction done.")

    # 2. Train model
    print("Starting training...")
    subprocess.run([
        "python",
        "train.py",
        "--epochs", str(args.epochs),
        "--outdir", args.outdir
    ], check=True)

    print("Pipeline complete.")

if __name__ == "__main__":
    main()