import os, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

    data = build_df_from_ravdess(DATA_ROOT)
    print(data.head(), len(data))

    train_idx, val_idx, test_idx = stratified_splits(
        data, label_col="Emotion", test_size=0.2, val_size=0.1, seed=SEED
    )
    np.save("splits/train_idx.npy", train_idx)
    np.save("splits/val_idx.npy",   val_idx)
    np.save("splits/test_idx.npy",  test_idx)

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

    m, s = compute_train_norm(train_loader, n_mfcc=N_MFCC, device=device)
    torch.save({"mean": m.cpu(), "std": s.cpu()}, os.path.join(OUTDIR, "norm.pt"))

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
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

        print(f"epoch {epoch+1:02d} | train loss {train_loss:.4f} | train acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "n_mfcc": N_MFCC, "n_classes": 8},
                os.path.join(OUTDIR, "best.ckpt"),
            )

    best_ckpt = torch.load(os.path.join(OUTDIR, "best.ckpt"), map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    test_loss, test_acc = eval_loop(model, test_loader, device, criterion, m, s)
    print(f"\nTEST | loss {test_loss:.4f} | acc {test_acc:.3f}")

if __name__ == "__main__":
    main()