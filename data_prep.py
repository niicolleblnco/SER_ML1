import argparse
import os
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--mfcc-path", type=str, required=True)
    args = ap.parse_args()

    print("Checking MFCC features...")

    if os.path.exists(args.mfcc_path):
        print("MFCCs already exist. Skipping extraction.")
        return

    os.makedirs(os.path.dirname(args.mfcc_path), exist_ok=True)

    print("Extracting MFCCs...")
    subprocess.run([
        "python", "utils.feature",
        "--data-root", args.data_root,
        "--out", args.mfcc_path
    ], check=True)

    if not os.path.exists(args.mfcc_path):
        raise RuntimeError("MFCC extraction failed.")

    print("MFCC extraction complete.")

if __name__ == "__main__":
    main()