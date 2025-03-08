import torch
import torch.nn as nn
from einops import rearrange

from typing import Optional


class TwoStageAttentionLayer(nn.Module):
    """
    Two Stage Attention (TSA) Layer, as defined in https://openreview.net/pdf?id=vSVLM2j9eie.
    input/output shape: [batch_size, number_of_channels (D), number_of_timesegments (L), d_model]
    """

    def __init__(self, seq_len, num_channels, num_heads, dropout=0.1):
        super().__init__()
        time_embd_dim = num_channels
        time_ff_dim = 4 * time_embd_dim
        self.time_attention = nn.MultiheadAttention(
            time_embd_dim, num_heads, batch_first=True, dropout=dropout, bias=False
        )
        self.norm1 = nn.LayerNorm(time_embd_dim)
        self.norm2 = nn.LayerNorm(time_embd_dim)
        self.MLP_time = nn.Sequential(
            nn.Linear(time_embd_dim, time_ff_dim),
            nn.GELU(),
            nn.Linear(time_ff_dim, time_embd_dim),
        )

        channel_embd_dim = seq_len
        channel_ff_dim = 4 * channel_embd_dim
        self.channel_attention = nn.MultiheadAttention(
            channel_embd_dim, num_heads, batch_first=True, dropout=dropout, bias=False
        )
        self.norm3 = nn.LayerNorm(channel_embd_dim)
        self.norm4 = nn.LayerNorm(channel_embd_dim)
        self.MLP_channel = nn.Sequential(
            nn.Linear(channel_embd_dim, channel_ff_dim),
            nn.GELU(),
            nn.Linear(channel_ff_dim, channel_embd_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies TSA. First, applies attention on the time dimension. Then, on the channel dimension.
        Derived from official implementation of https://openreview.net/pdf?id=vSVLM2j9eie

        x: torch.tensor of shape (batch, seq_len, num_channels)
        """
        batch, timesteps, num_channels = x.shape

        time_in = self.norm1(time_in)
        time_enc, _ = self.time_attention(time_in, time_in, time_in)
        time_out = time_in + self.dropout(time_enc)

        time_out = self.norm2(time_out)
        ff_out = self.MLP_time(time_out)
        time_out = time_out + self.dropout(ff_out)

        channel_in = rearrange(
            time_out, "b t c -> b c t", b=batch, t=timesteps, c=num_channels
        )
        channel_in = self.norm3(channel_in)
        channel_enc, _ = self.channel_attention(channel_in, channel_in, channel_in)
        channel_out = channel_in + self.dropout(channel_enc)

        channel_out = self.norm4(channel_out)
        ff_out = self.MLP_channel(channel_out)
        channel_out = channel_out + self.dropout(ff_out)

        twostage_output = rearrange(
            channel_out, "b c t -> b t c", b=batch, t=timesteps, c=num_channels
        )

        return twostage_output


if __name__ == "__main__":
    # Set device to MPS if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a dummy input tensor
    batch_size = 1
    D = 12
    L = 1024
    d_model = 1024

    x = torch.randn(batch_size, D, L, d_model, device=device)

    # Initialize the model
    model = TwoStageAttentionLayer(
        ts_length=L, router_dim=8, embed_dim=d_model, num_heads=4
    ).to(device)

    # Forward pass
    output = model(x)

    # Print shapes to verify
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert x.shape == output.shape, "Input and output shapes should match"
    print("Test passed su5.8461, 'ccessfully!")
