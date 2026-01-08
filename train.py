import os
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.dataset import PrecomputedMFCCDataset
from utils.collate_transformer import collate_transformer_train, collate_transformer_eval
from utils.split import stratified_splits
from transformer_model import SmallTransformerSER

# Number of emotion classes in your label mapping
N_CLASSES = 8


def get_args():
    """
    Defines and parses command-line arguments for training.
    """
    import argparse
    ap = argparse.ArgumentParser()

    # Training loop parameters
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="runs/exp1")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Input features path
    ap.add_argument("--mfcc_path", type=str, default="data/features/precomputed_mfcc.npz")

    # Reproducibility and checkpoint history
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_history", type=int, default=5)

    # Training-time cropping length (frames)
    ap.add_argument("--chunk_len", type=int, default=300)

    # Model hyperparameters
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=5)
    ap.add_argument("--ff_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    return ap.parse_args()


def eval_loop(model, loader, device, criterion):
    """
    Runs evaluation (validation or test) over one dataloader.

    Returns
    -------
    avg_loss : float
        Mean loss over the dataset.
    avg_acc : float
        Accuracy over the dataset.
    """
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, mask, yb in tqdm(loader, desc="eval", leave=False):
            # Move batch to device
            xb = xb.to(device)
            mask = mask.to(device)
            yb = yb.to(device)

            # Forward pass
            logits = model(xb, mask)

            # Loss and metrics
            loss = criterion(logits, yb)
            bs = int(xb.size(0))

            loss_sum += float(loss) * bs
            preds = logits.argmax(1)
            correct += int((preds == yb).sum().item())
            total += bs

    # Protect against divide-by-zero if loader is empty
    return (loss_sum / max(total, 1)), (correct / max(total, 1))


def safe_save(obj, path):
    """
    Atomic checkpoint save:
    writes to a temporary file and then replaces the target file.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    # Parse args and create output directory
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # TensorBoard logging directory inside the run folder
    tensorboard_dir = os.path.join(args.outdir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Save config for reproducibility
    config_path = os.path.join(args.outdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Seed all RNGs used by your pipeline
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Select training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MFCC dataset
    if not os.path.exists(args.mfcc_path):
        raise FileNotFoundError(f"MFCC file not found: {args.mfcc_path}")

    dataset = PrecomputedMFCCDataset(args.mfcc_path)

    # Convert labels to numpy array for splitting
    labels = np.array(dataset.labels, dtype=np.int64)

    # Stratified splits preserve class proportions
    idx_train, idx_val, idx_test = stratified_splits(
        labels, test_size=0.2, val_size=0.1, seed=args.seed
    )

    # Wrap subsets with indices
    train_set = Subset(dataset, idx_train)
    val_set = Subset(dataset, idx_val)
    test_set = Subset(dataset, idx_test)

    # Torch generator used by DataLoader (e.g. shuffle reproducibility)
    g = torch.Generator()
    g.manual_seed(args.seed)

    def _worker_init_fn(worker_id):
        """
        Seeds NumPy and Python random per worker.
        Important because your training collate uses Python random for cropping.
        """
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)

    # Training loader uses random cropping
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_transformer_train(b, chunk_len=args.chunk_len),
        worker_init_fn=_worker_init_fn,
        generator=g,
        num_workers=0
    )

    # Validation and test loaders keep full sequences
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_transformer_eval,
        worker_init_fn=_worker_init_fn,
        generator=g,
        num_workers=0
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_transformer_eval,
        worker_init_fn=_worker_init_fn,
        generator=g,
        num_workers=0
    )

    # Infer MFCC feature dimension from one sample
    sample_x, _ = dataset[0]
    n_mfcc = int(sample_x.shape[1])

    # Build model
    model = SmallTransformerSER(
        n_mfcc=n_mfcc,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        n_classes=N_CLASSES,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    # Optimizer, scheduler, loss
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # TensorBoard logging
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Directory storing multiple best checkpoints over time
    history_dir = os.path.join(args.outdir, "versions")
    os.makedirs(history_dir, exist_ok=True)

    best_ckpt_path = os.path.join(args.outdir, "best.ckpt")

    # State for resume and best tracking
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = None

    # Resume if a best checkpoint exists
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"])
        best_val_acc = float(ckpt["val_acc"])
        best_val_loss = float(ckpt.get("val_loss", 1e9))

    # Main training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for xb, mask, yb in tqdm(train_loader, desc=f"epoch {epoch+1} train"):
            xb = xb.to(device)
            mask = mask.to(device)
            yb = yb.to(device)

            # Forward
            logits = model(xb, mask)
            loss = criterion(logits, yb)

            # Backprop
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Metrics
            bs = int(xb.size(0))
            running_loss += float(loss) * bs
            preds = logits.argmax(1)
            correct += int((preds == yb).sum().item())
            total += bs

        # Aggregate training metrics
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation and scheduler update
        val_loss, val_acc = eval_loop(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        # Current learning rate (after scheduler step)
        lr_now = opt.param_groups[0]["lr"]

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        # Console logging
        print(
            "epoch", epoch + 1,
            "| train loss", f"{train_loss:.4f}",
            "| train acc", f"{train_acc:.3f}",
            "| val loss", f"{val_loss:.4f}",
            "| val acc", f"{val_acc:.3f}",
            "| lr", f"{lr_now:.2e}"
        )

        # Save checkpoint only when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)

            best_model = {
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": best_val_loss,
                "epoch": epoch + 1
            }
            safe_save(best_model, best_ckpt_path)

            # Keep a history snapshot for later inspection
            hist_path = os.path.join(
                history_dir,
                f"best_epoch_{epoch+1}_acc_{best_val_acc:.3f}.pt"
            )
            safe_save(best_model, hist_path)

            # Maintain only the most recent max_history snapshots
            history_files = sorted(
                [f for f in os.listdir(history_dir) if f.endswith(".pt")]
            )
            if len(history_files) > args.max_history:
                for old_file in history_files[:-args.max_history]:
                    os.remove(os.path.join(history_dir, old_file))

    # Load best checkpoint and evaluate on test set
    best = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best["state_dict"])

    test_loss, test_acc = eval_loop(model, test_loader, device, criterion)
    print("TEST | loss", f"{test_loss:.4f}", "| acc", f"{test_acc:.3f}")

    # Append run results into a single CSV under the parent of outdir
    results_path = os.path.join(os.path.dirname(args.outdir), "results.csv")
    write_header = not os.path.exists(results_path)

    with open(results_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "run_name",
                "lr",
                "batch_size",
                "best_val_acc",
                "best_val_loss",
                "test_acc",
                "test_loss"
            ])
        w.writerow([
            os.path.basename(args.outdir),
            args.lr,
            args.batch_size,
            best_val_acc,
            best_val_loss if best_val_loss is not None else "",
            test_acc,
            test_loss
        ])

    writer.close()


if __name__ == "__main__":
    main()