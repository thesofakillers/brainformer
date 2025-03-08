import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

from typing import Optional


class TwoStageAttentionLayer(nn.Module):
    """
    Two Stage Attention (TSA) Layer, as defined in https://openreview.net/pdf?id=vSVLM2j9eie.
    input/output shape: [batch_size, number_of_channels (D), number_of_timesegments (L), d_model]
    """

    def __init__(
        self, ts_length, router_dim, embed_dim, num_heads, ff_dim=None, dropout=0.1
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.time_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout, bias=False
        )

        self.time_to_router_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout, bias=False
        )

        self.router_to_dimension_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout, bias=False
        )

        self.router = nn.Parameter(torch.randn(ts_length + 1, router_dim, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.MLP1 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask_time: Optional[torch.Tensor] = None,
        key_padding_mask_time: Optional[torch.Tensor] = None,
        attention_mask_dimension: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Applies TSA. First, applies attention on the time dimension. Then, on the channel dimension.
        Derived from official implementation of https://openreview.net/pdf?id=vSVLM2j9eie
        """
        batch, timesteps, channels, embed_dim = x.shape

        # Cross Time Stage: Apply MSA across time
        time_in = rearrange(
            x,
            "b ts_length ts_dim d_model -> (b ts_dim) ts_length d_model",
        )

        # self-attend all tokens time wise
        time_enc, _ = self.time_attention(
            time_in,
            time_in,
            time_in,
            attn_mask=attention_mask_time,
            key_padding_mask=key_padding_mask_time,
        )

        time_out = time_in + self.dropout(time_enc)
        time_out = self.norm1(time_out)
        ff_out = self.MLP1(time_out)
        time_out = time_out + self.dropout(ff_out)
        time_out = self.norm2(time_out)

        # Cross Dimension Stage: use a small router matrix and then distribute messages to build D-to-D connections
        dimension_in = rearrange(
            time_out,
            "(b ts_dim) ts_length d_model -> (b ts_length) ts_dim d_model",
            b=batch,
            ts_dim=channels,
            ts_length=timesteps,
            d_model=embed_dim,
        )
        batch_router = repeat(
            self.router,
            "ts_length router_dim d_model -> (repeat ts_length) router_dim d_model",
            repeat=batch,
            ts_length=timesteps,
        )

        # dimensions are never padded, so no key_padding_mask_dimension
        dimension_buffer, _ = self.time_to_router_attention(
            batch_router, dimension_in, dimension_in
        )

        dimension_received_router, _ = self.router_to_dimension_attention(
            dimension_in,
            dimension_buffer,
            dimension_buffer,
        )

        dimension_out = dimension_in + self.dropout(dimension_received_router)
        dimension_out = self.norm3(dimension_out)
        ff_out = self.MLP2(dimension_out)
        dimension_out = dimension_out + self.dropout(ff_out)
        dimension_out = self.norm4(dimension_out)

        twostage_output = rearrange(
            dimension_out,
            "(b ts_length) ts_dim d_model -> b ts_length ts_dim d_model",
            b=batch,
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
    print("Test passed successfully!")
