import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for RotatoryPostionalEncoding")
        
        self.d_model = d_model
        self.max_len = max_len

        dim = torch.arange(0, d_model, 2, dtype=torch.float32)

        inv_freq = 1.0 /(1000 ** (dim / d_model))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_sin_cos(self, seq_len, device):
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        sin = freqs.sin().unsqueeze(0)
        cos = freqs.cos().unsqueeze(0)

        return sin, cos
    
    def forward(self, x):
        T = x.shape[1]
        sin, cos = self.get_sin_cos(T, x.device)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        
        out  = torch.zeros_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd

        return out

class SmallTransformerSER(nn.Module):
    def __init__(self, n_mfcc=20, d_model=128, n_heads=4, num_layers=2, n_classes=8):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for RoPE")
        
        self.input_proj = nn.Linear(n_mfcc, d_model) 
        self.rope = RotaryPositionalEncoding(d_model) 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, x, mask):
        x = self.input_proj(x)
        x = self.rope(x)
        encoded = self.transformer(x, src_key_padding_mask=mask)
        pooled = encoded.mean(dim=1)

        return self.classifier(pooled)