"""
A transformer encoder block. The encoder is used to preprocess the EEG data.

For instance, you could add layers that output a matrix of shape (batch_size * num_heads, seq_length, seq_length)
that would be used as a custom attention mask for the self-attention layer.
Ultimately, attention is ~a convolution kernel, so you could as well learn it.
"""

import torch
import torch.nn as nn
from typing import Optional


class EncoderLayer(nn.Module):
    """
    A transformer encoder layer, implementing pre-normalization with dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional mask of shape (seq_len, seq_len) or
                            (N*num_heads, seq_len, seq_len)
            key_padding_mask: Optional mask of shape (batch, seq_len) where True indicates padding.
                              Thus, that token is not attended.
        """
        # Pre-norm architecture (more stable training)
        x = self.layernorm1(x)
        # Self-attention with both masks
        attention_out, _ = self.self_attention(
            x, x, x, attn_mask=attention_mask, key_padding_mask=key_padding_mask
        )
        # Apply dropout and add residual connection
        x = x + self.dropout1(attention_out)

        # Pre-norm for feed-forward
        x = self.layernorm2(x)
        ff_out = self.feed_forward(x)
        # Apply dropout and add residual connection
        x = x + self.dropout2(ff_out)

        return x


class EncoderBlock(nn.Module):
    """
    A transformer encoder block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional mask of shape (L, S) or (N*num_heads, L, S)
            key_padding_mask: Optional mask of shape (batch, seq_len) where True indicates padding
        """
        # Pass through all encoder layers with batch_first=True
        for layer in self.layers:
            x = layer(x, attention_mask, key_padding_mask)

        return x
