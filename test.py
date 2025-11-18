import torch
from model import masked_mean, MFCCMeanLinear

B, C, T = 8, 20, 100
x = torch.randn(B, C, T)
lengths = torch.randint(low=50, high=T+1, size=(B,))
m = MFCCMeanLinear(n_mfcc=C, n_classes=8)
out = m(x, lengths)
print(out.shape)  


