import os, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from transformer_model import SmallTransformerSER
from root.utils.collate_transformer import collate_transformer_train, collate_transformer_eval
from root.utils.split import stratified_splits

EPOCHS      = 40
BATCH_SIZE  = 8
LR          = 1e-3
SEED        = 42
N_CLASSES = 8

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--outdir", type=str, default="runs/exp1")
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

    torch.manual_seed(SEED)
    os.makedirs(args.outdir, exist_ok=True)
    history_dir = os.path.join(args.outdir, "versions")
    os.makedirs(history_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.outdir)

    MFCC_PATH = "data/features/ravdess/precomputed_mfcc.npz"
    if not os.path.exists(MFCC_PATH):
        print("precomputed_mfcc.npz not found. Extracting MFCC sequences...")
    data = np.load(MFCC_PATH, allow_pickle=True)
    mfcc_list = torch.tensor(data["mfccs"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    
    idx_train, idx_val, idx_test = stratified_splits(
        labels, test_size=0.2, val_size=0.1, seed=SEED
    )

    train_data = [(mfcc_list[i], int(labels[i])) for i in idx_train]
    val_data   = [(mfcc_list[i], int(labels[i])) for i in idx_val]
    test_data  = [(mfcc_list[i], int(labels[i])) for i in idx_test]

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_transformer_train
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

    best_ckpt_path = os.path.join(args.outdir, "best.ckpt")

    if os.path.exists(best_ckpt_path):
        print("Loading previous best checkpoint...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_acc = ckpt["val_acc"]
        print(f"Continuing training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_acc = 0.0

    best_val_acc = 0.0

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
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
        
        TARGET_ACC = 0.6
        if val_acc >= TARGET_ACC:
            print("Target accuracy reached, stopping early.")
            torch.save(
                model.state_dict(),
                os.path.join(args.outdir, "early_stop.pt")
            )
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = {
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch + 1
            }
            torch.save(best_model, os.path.join(args.outdir, "best.ckpt"))

            hist_path = os.path.join(history_dir, f"best_epoch_{epoch+1}_acc_{val_acc:.3f}.pt")
            torch.save(best_model, hist_path)

    best = torch.load(os.path.join(args.outdir, "best.ckpt"), map_location=device)
    model.load_state_dict(best["state_dict"])

    test_loss, test_acc = eval_loop(model, test_loader, device, criterion)
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc:.3f}")

    writer.close()

if __name__ == "__main__":
    main()