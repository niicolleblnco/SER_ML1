import argparse
import subprocess
import os



def main(data_root, epochs=40, outdir="temp"):
    for i in range(epochs):



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="runs/exp1")
    args = ap.parse_args()
    yaml.dump(file, args)
    main(**args)
 