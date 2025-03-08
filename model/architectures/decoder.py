"""
A transformer decoder block. The decoder is used to autoregressively predict MEG data, conditioned on MEG itself and the
encoder's processing of EEG data.

For instance, you could add layers that output a matrix of shape (batch_size * num_heads, seq_length, seq_length)
that would be used as a custom attention mask for the self-attention layer.
Ultimately, attention is ~a convolution kernel, so you could as well learn it.
"""

import torch
import torch.nn as nn
from typing import Optional


class DecoderLayer(nn.Module):
    """
    A decoder layer, implementing pre-normalization with dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Self attention layer
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )

        # Cross attention layer
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )

        # Feed forward layer
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        cross_key: Optional[torch.Tensor] = None,
        cross_value: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        self_padding_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_padding_mask: Optional[torch.Tensor] = None,
    ):
        # Self attention
        x_norm = self.layernorm1(x)

        self_attn_output, _ = self.self_attention(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=self_attention_mask,
            key_padding_mask=self_padding_mask,
        )
        x = x + self.dropout1(self_attn_output)

        # Cross attention
        x_norm = self.layernorm2(x)
        cross_attn_output, _ = self.cross_attention(
            x_norm,
            cross_key,
            cross_value,
            attn_mask=cross_attention_mask,
            key_padding_mask=cross_padding_mask,
        )
        x = x + cross_attn_output

        # Feed forward
        x_norm = self.layernorm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout2(ff_output)

        return x


class DecoderBlock(nn.Module):
    """
    A decoder block, implementing pre-normalization with dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        n_layers: int,
    ):
        super().__init__()

        # Create stack of decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        self_padding_mask: Optional[torch.Tensor] = None,
        cross_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder block.

        Args:
            x: Input tensor of shape (batch_size, target_seq_len, embed_dim)
            encoder_output: Output from the encoder of shape (batch_size, source_seq_len, embed_dim)
            self_attention_mask: Causal mask for decoder self-attention
            cross_attention_mask: Optional mask for cross-attention with encoder
            self_padding_mask: Mask for padding in decoder sequence
            cross_padding_mask: Mask for padding in encoder sequence

        Returns:
            Tensor of shape (batch_size, target_seq_len, embed_dim)
        """
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(
                x,
                cross_key=encoder_output,
                cross_value=encoder_output,
                attention_mask=self_attention_mask,
                key_padding_mask=self_padding_mask,
                cross_attention_mask=cross_attention_mask,
                cross_padding_mask=cross_padding_mask,
            )

        return x
