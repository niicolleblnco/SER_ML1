import argparse
import subprocess
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--exp-root", type=str, default="experiments")
    args = ap.parse_args()

    MFCC_PATH = "data/features/precomputed_mfcc.npz"

    # 1. Ensure data exists (automatic, safe)
    subprocess.run([
        "python", "data_prep.py",
        "--data-root", args.data_root,
        "--mfcc-path", MFCC_PATH
    ], check=True)

    # 2. Sweep
    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [8, 16, 32]

    for lr in learning_rates:
        for bs in batch_sizes:
            if lr == 1e-3 and bs == 32:
                continue

            outdir = os.path.join(args.exp_root, f"lr{lr}_bs{bs}")
            os.makedirs(outdir, exist_ok=True)

            subprocess.run([
                "python", "-m", "ser_ml1.train",
                "--mfcc_path", MFCC_PATH,
                "--lr", str(lr),
                "--batch_size", str(bs),
                "--epochs", str(args.epochs),
                "--outdir", outdir
            ], check=True)

if __name__ == "__main__":
    main()