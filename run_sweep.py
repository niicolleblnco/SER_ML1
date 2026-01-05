import argparse
import subprocess
import os

def main():
    # Parse sweep configuration
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--exp-root", type=str, default="experiments")
    args = ap.parse_args()

    # Fixed location for precomputed MFCC features
    MFCC_PATH = "data/features/precomputed_mfcc.npz"

    # Step 1. Ensure MFCC features exist
    # This is a hard dependency. The sweep stops if this fails.
    subprocess.run(
        [
            "python", "data_prep.py",
            "--data-root", args.data_root,
            "--mfcc-path", MFCC_PATH
        ],
        check=True
    )

    # Step 2. Hyperparameter grids
    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [8, 16, 32]
    num_layers_list = [3, 5, 7]
    dropout_list = [0.1, 0.2, 0.3]

    # Step 3. Run sweep
    for lr in learning_rates:
        for bs in batch_sizes:
            for num_layers in num_layers_list:
                for dropout in dropout_list:
                    # Skip a known unstable or unwanted configuration
                    if lr == 1e-3 and bs == 32:
                        continue

                    # Build run name and output directory
                    run_name = f"lr{lr}_bs{bs}_layers{num_layers}_do{dropout}"
                    outdir = os.path.join(args.exp_root, run_name)
                    os.makedirs(outdir, exist_ok=True)

                    # Launch training process
                    subprocess.run(
                        [
                            "python", "train.py",
                            "--mfcc_path", MFCC_PATH,
                            "--lr", str(lr),
                            "--batch_size", str(bs),
                            "--epochs", str(args.epochs),
                            "--num_layers", str(num_layers),
                            "--dropout", str(dropout),
                            "--outdir", outdir
                        ],
                        check=True
                    )

if __name__ == "__main__":
    main()