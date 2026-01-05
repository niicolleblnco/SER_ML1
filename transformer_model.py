import argparse
import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).

    Applies a deterministic rotation to pairs of feature dimensions
    based on token position, allowing relative positional information
    to be encoded directly into the embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()

        # RoPE requires an even embedding dimension
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for RotaryPositionalEncoding")

        # Frequencies for each pair of dimensions
        dim = torch.arange(0, d_model, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (dim / d_model))

        # Register as buffer so it moves with the model but is not trainable
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_sin_cos(self, seq_len: int, device):
        """
        Computes sine and cosine positional embeddings.

        Returns tensors shaped (1, T, d_model / 2) for broadcasting.
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        sin = freqs.sin().unsqueeze(0)
        cos = freqs.cos().unsqueeze(0)
        return sin, cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary positional encoding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, D), where D must be even.

        Returns
        -------
        torch.Tensor
            Position-encoded tensor of the same shape.
        """
        T = x.shape[1]
        sin, cos = self.get_sin_cos(T, x.device)

        # Split embedding into even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Re-interleave dimensions
        out = torch.zeros_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out


class SmallTransformerSER(nn.Module):
    """
    Transformer-based Speech Emotion Recognition model.

    Uses MFCC features as input, rotary positional encoding,
    and mean pooling over time with padding awareness.
    """
    def __init__(
        self,
        n_mfcc: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 5,
        n_classes: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Transformer embedding dimension must be even for RoPE
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        # Project MFCC features to transformer embedding space
        self.input_proj = nn.Linear(n_mfcc, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Rotary positional encoding
        self.rope = RotaryPositionalEncoding(d_model)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.pre_classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, n_mfcc).
        mask : torch.Tensor
            Shape (B, T). True indicates padding positions.

        Returns
        -------
        torch.Tensor
            Shape (B, n_classes), class logits.
        """
        # Input projection and regularization
        x = self.input_proj(x)
        x = self.input_dropout(x)

        # Apply rotary positional encoding
        x = self.rope(x)

        # Transformer encoder
        encoded = self.transformer(
            x,
            src_key_padding_mask=mask
        )

        # Mask-aware mean pooling over time
        valid = (~mask).unsqueeze(-1).float()
        encoded = encoded * valid
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = encoded.sum(dim=1) / denom

        # Classification head
        pooled = self.pre_classifier_dropout(pooled)
        return self.classifier(pooled)


def build_argparser():
    """
    Argument parser for inspecting or instantiating the model.
    """
    p = argparse.ArgumentParser(description="Small Transformer SER with RoPE")
    p.add_argument("--n_mfcc", type=int, default=20)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--n_classes", type=int, default=8)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    return p


def main():
    # Instantiate and print the model for inspection
    args = build_argparser().parse_args()
    model = SmallTransformerSER(
        n_mfcc=args.n_mfcc,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        n_classes=args.n_classes,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    print(model)


if __name__ == "__main__":
    main()