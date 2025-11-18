import torch
import torch.nn as nn

def masked_mean(x: torch.Tensor, lengths: torch.Tensor):
    B, C, T = x.shape
    device = x.device
    t = torch.arange(T, device=device)[None, :]
    mask = (t < lengths[:, None]).float()
    mask = mask[:, None, :]
    denom = mask.sum(dim=2).clamp_min(1.0)
    mean = (x * mask).sum(dim=2) / denom
    return mean

class MFCCMeanLinear(nn.Module):
    def __init__(self, n_mfcc: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_mfcc, n_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
            feats = masked_mean(x, lengths)
            logits = self.fc(feats)
            return logits
