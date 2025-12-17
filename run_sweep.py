import argparse
import subprocess
import os
import yaml


def main(data_root, epochs, outdir):
    os.makedirs(outdir, exist_ok=True)

    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [8, 16, 32]

    sweep = []

    for lr in learning_rates:
        for bs in batch_sizes:
            if lr == 1e-3 and bs == 32:
                continue
            sweep.append({"lr": lr, "batch_size": bs})

    for cfg in sweep:
        name = f"lr{cfg['lr']}_bs{cfg['batch_size']}"
        cfg_path = os.path.join(outdir, f"{name}.yaml")

        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        subprocess.run([
            "python",
            "train.py",
            "--data-root", data_root,
            "--epochs", str(epochs),
            "--config", cfg_path
        ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="sweep_configs")
    args = ap.parse_args()

    main(args.data_root, args.epochs, args.outdir)