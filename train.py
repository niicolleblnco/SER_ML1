import os, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import subprocess


from transformer_model import SmallTransformerSER
from collate_transformer import collate_transformer, collate_transformer_eval
from split import stratified_splits

EPOCHS      = 40
BATCH_SIZE  = 8
LR          = 1e-3
OUTDIR      = "runs/exp_transformer1"
SEED        = 42
N_CLASSES = 8

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    return ap.parse_args()


def eval_loop(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, mask, yb in tqdm(loader, desc="eval", leave=False):
            xb = xb.to(device)
            mask = mask.to(device)
            yb = yb.to(device)
            
            logits = model(xb, mask)
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
    os.makedirs("checkpoints_tr1", exist_ok=True)
    os.makedirs("checkpoints_tr1/versions", exist_ok=True)
    writer = SummaryWriter(log_dir=OUTDIR)

    if not os.path.exists("precomputed_mfcc.npz"):
        print("precomputed_mfcc.npz not found. Extracting MFCC sequences...")
        subprocess.run(["python", "extract_mfcc_sequence.py"], check=True)
    data = np.load("precomputed_mfcc.npz", allow_pickle=True)
    mfcc_list = data["mfccs"]
    labels = data["labels"]
    
    idx_train, idx_val, idx_test = stratified_splits(
        labels, label_col=None, test_size=0.2, val_size=0.1, seed=SEED
    )

    train_data = [(mfcc_list[i], int(labels[i])) for i in idx_train]
    val_data   = [(mfcc_list[i], int(labels[i])) for i in idx_val]
    test_data  = [(mfcc_list[i], int(labels[i])) for i in idx_test]

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_transformer
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_transformer_eval
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_transformer_eval
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_mfcc = mfcc_list[0].shape[1]
    model = SmallTransformerSER(n_mfcc=n_mfcc, n_classes=N_CLASSES).to(device)

    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    checkpoint_latest = "checkpoints_tr1/latest.pt"
    start_epoch = 0

    if os.path.exists(checkpoint_latest):
        print("Resuming training from latest checkpoint...")
        ckpt = torch.load(checkpoint_latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    best_val_acc = 0.0

    for epoch in range(start_epoch, total_epochs_requested):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, mask, yb in tqdm(train_loader, desc=f"epoch {epoch+1} train"):
            xb = xb.to(device)
            mask = mask.to(device)
            yb = yb.to(device)

            logits = model(xb, mask)
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
        val_loss, val_acc = eval_loop(model, val_loader, device, criterion)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        print(f"epoch {epoch+1} | "
              f"train loss {train_loss:.4f} | train acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.3f}")
        
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
                "val_acc": val_acc,
                "epoch": epoch + 1
            }

            torch.save(best_model, os.path.join(OUTDIR, "best.ckpt"))

            hist_path = f"checkpoints_tr1/versions/best_epoch_{epoch+1}_acc_{val_acc:.3f}.pt"
            torch.save(best_model, hist_path)

    best = torch.load(os.path.join(OUTDIR, "best.ckpt"), map_location=device)
    model.load_state_dict(best["state_dict"])

    test_loss, test_acc = eval_loop(model, test_loader, device, criterion)
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc:.3f}")

    

    writer.close()

if __name__ == "__main__":
    main()