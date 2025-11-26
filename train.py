import os, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import subprocess

from model import MFCCMeanLinear
from split import stratified_splits

EPOCHS      = 40
BATCH_SIZE  = 8
LR          = 1e-3
OUTDIR      = "runs/exp1"
SEED        = 42
N_CLASSES = 8

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    return ap.parse_args()


def load_or_extract_features():
    feats_path = "features.npy"
    labs_path = "labels.npy"

    if os.path.exists(feats_path) and os.path.exists(labs_path):
        print("Loading existing precomputed features...")
        feats = np.load(feats_path)
        labs = np.load(labs_path)
        return feats, labs

    print("No precomputed features found. Running feature_extraction.py...")
    subprocess.run(["python", "feature_extraction.py"], check=True)

    feats = np.load(feats_path)
    labs = np.load(labs_path)
    return feats, labs

def compute_norm(feats):
    m = feats.mean(axis=0)
    s = feats.std(axis=0).clip(min=1e-8)
    return m, s

def eval_loop(model, loader, device, criterion, m, s):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="eval", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            feats = (xb - m) / s
            logits = model.fc(feats)
            loss = criterion(logits, yb)

            loss_sum += float(loss) * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return (loss_sum / total), (correct / total)

def main():
    args = get_args()
    total_epochs_requested = args.epochs

    torch.manual_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)
    writer = SummaryWriter(log_dir=OUTDIR)

    feats, labs = load_or_extract_features()
    feats = feats.astype(np.float32)
    labs = labs.astype(np.int64)
    
    idx_train, idx_val, idx_test = stratified_splits(
        labs, label_col=None, test_size=0.2, val_size=0.1, seed=SEED
    )

    x_train, y_train = feats[idx_train], labs[idx_train]
    x_val,   y_val   = feats[idx_val],   labs[idx_val]
    x_test,  y_test  = feats[idx_test],  labs[idx_test]

    m, s = compute_norm(x_train)
    m = torch.tensor(m, dtype=torch.float32)
    s = torch.tensor(s, dtype=torch.float32)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(x_val),   torch.tensor(y_val))
    test_ds  = TensorDataset(torch.tensor(x_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    n_mfcc = feats.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MFCCMeanLinear(n_mfcc=n_mfcc, n_classes=N_CLASSES).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/versions", exist_ok=True)

    checkpoint_latest = "checkpoints/latest.pt"
    start_epoch = 0

    if os.path.exists(checkpoint_latest):
        ckpt = torch.load(checkpoint_latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]

    end_epoch = start_epoch + total_epochs_requested

    best_val_acc = 0.0

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch+1} train"):
            xb, yb = xb.to(device), yb.to(device)
            feats = (xb - m) / s
            logits = model.fc(feats)
            loss = criterion(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss) * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total
        val_loss, val_acc = eval_loop(model, val_loader, device, criterion, m, s)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        print(f"epoch {epoch+1} | train loss {train_loss:.4f} | train acc {train_acc:.3f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict()
        }
        torch.save(checkpoint, checkpoint_latest)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = {
                "state_dict": model.state_dict(),
                "n_mfcc": n_mfcc,
                "n_classes": 8,
                "epoch": epoch + 1,
                "val_acc": val_acc
            }
            torch.save(best_model, os.path.join(OUTDIR, "best.ckpt"))
            hist_path = f"checkpoints/versions/best_epoch_{epoch+1}_acc_{val_acc:.3f}.pt"
            torch.save(best_model, hist_path)

    writer.close()
    best_ckpt = torch.load(os.path.join(OUTDIR, "best.ckpt"), map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    test_loss, test_acc = eval_loop(model, test_loader, device, criterion, m, s)
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc:.3f}")

if __name__ == "__main__":
    main()