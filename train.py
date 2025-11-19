import os, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import MFCCMeanLinear, masked_mean
from features import SERAudioDataset, collate_mfcc
from loady import build_df_from_ravdess  
from split import stratified_splits

DATA_ROOT   = "data"
EPOCHS      = 40
BATCH_SIZE  = 8
LR          = 1e-3
SAMPLE_RATE = 16000
N_MFCC      = 20
NUM_WORKERS = 0
OUTDIR      = "runs/exp1"
SEED        = 42

def compute_train_norm(train_loader, n_mfcc: int, device: torch.device):
    sum_, sum2, count = torch.zeros(n_mfcc), torch.zeros(n_mfcc), 0
    with torch.no_grad():
        for xb, lengths, yb, _ in train_loader:
            f = masked_mean(xb, lengths)     
            sum_  += f.sum(0)
            sum2  += (f ** 2).sum(0)
            count += f.size(0)
    feat_mean = sum_ / count
    feat_var  = (sum2 / count) - feat_mean**2
    feat_std  = feat_var.clamp_min(1e-8).sqrt()
    return feat_mean.to(device), feat_std.to(device)

def eval_loop(model, loader, device, criterion, m, s):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, lengths, yb, _ in loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            feats = masked_mean(xb, lengths)
            feats = (feats - m) / s
            logits = model.fc(feats)
            loss = criterion(logits, yb)
            loss_sum += float(loss) * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return (loss_sum / total), (correct / total)

def main():
    torch.manual_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs("splits", exist_ok=True)
    writer = SummaryWriter(log_dir=OUTDIR)

    df_cache = "splits/data_df.npy"

    if os.path.exists(df_cache):
        data = np.load(df_cache, allow_pickle=True).item()
        data = data["df"]
    else:
        df = build_df_from_ravdess(DATA_ROOT)
        np.save(df_cache, {"df": df})
        data = df

    split_cache = "splits/split_indices.npz"

    if os.path.exists(split_cache):
        split_data = np.load(split_cache)
        train_idx = split_data["train"]
        val_idx   = split_data["val"]
        test_idx  = split_data["test"]
    else:
        train_idx, val_idx, test_idx = stratified_splits(
        data, label_col="Emotion", test_size=0.2, val_size=0.1, seed=SEED
        )
        np.savez(split_cache, train=train_idx, val=val_idx, test=test_idx)

    train_df = data.loc[train_idx].reset_index(drop=True)
    val_df   = data.loc[val_idx].reset_index(drop=True)
    test_df  = data.loc[test_idx].reset_index(drop=True)

    train_ds = SERAudioDataset(train_df, sample_rate=SAMPLE_RATE, feature_type="mfcc", n_mfcc=N_MFCC)
    val_ds   = SERAudioDataset(val_df,   sample_rate=SAMPLE_RATE, feature_type="mfcc", n_mfcc=N_MFCC)
    test_ds  = SERAudioDataset(test_df,  sample_rate=SAMPLE_RATE, feature_type="mfcc", n_mfcc=N_MFCC)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_mfcc, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mfcc, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mfcc, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFCCMeanLinear(n_mfcc=N_MFCC, n_classes=8).to(device) # if statement
    
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    checkpoint_latest = "checkpoints/latest.pt"
    checkpoint_versions = "checkpoints/versions"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(checkpoint_versions, exist_ok=True)

    start_epoch = 0

    if os.path.exists(checkpoint_latest):
        ckpt = torch.load(checkpoint_latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]

    
    EPOCHS_PER_RUN = 10
    end_epoch = start_epoch + EPOCHS_PER_RUN

    m, s = compute_train_norm(train_loader, n_mfcc=N_MFCC, device=device)
    torch.save({"mean": m.cpu(), "std": s.cpu()}, os.path.join(OUTDIR, "norm.pt"))

    best_val_acc = 0.0
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for xb, lengths, yb, _ in train_loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            feats = masked_mean(xb, lengths)
            feats = (feats - m) / s
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

        print(f"epoch {epoch+1:02d} | train loss {train_loss:.4f} | train acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.3f}")
        
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict()
        }

        torch.save(checkpoint, checkpoint_latest)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = {
                "state_dict": model.state_dict(),
                "n_mfcc": N_MFCC,
                "n_classes": 8,
                "epoch": epoch + 1,
                "val_acc": val_acc
            }

            # overwrite the main best ckpt
            torch.save(best_model, os.path.join(OUTDIR, "best.ckpt"))

            # save all historical bests
            hist_path = os.path.join(checkpoint_versions, f"best_epoch_{epoch+1}_acc_{val_acc:.3f}.pt")
            torch.save(best_model, hist_path)

    writer.close()
    best_ckpt = torch.load(os.path.join(OUTDIR, "best.ckpt"), map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    test_loss, test_acc = eval_loop(model, test_loader, device, criterion, m, s)
    print(f"\nTEST | loss {test_loss:.4f} | acc {test_acc:.3f}")

if __name__ == "__main__":
    main()