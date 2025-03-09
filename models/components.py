from einops import rearrange
import torch
import torch.nn as nn

from models.twostage_attention import TwoStageAttentionLayer


class EncoderBlock(nn.Module):
    def __init__(self, seq_len: int, num_channels: int, num_heads: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TwoStageAttentionLayer(seq_len, num_channels, num_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: torch.tensor of shape (batch, seq_len, src_channels)

        returns:
            torch.tensor of shape (batch, seq_len, src_channels)
        """
        for layer in self.layers:
            x = layer(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, seq_len, src_channels, tgt_channels, num_heads, n_layers):
        super().__init__()
        self.self_attn_layers = nn.ModuleList(
            [
                TwoStageAttentionLayer(seq_len, tgt_channels, num_heads, is_causal=True)
                for _ in range(n_layers)
            ]
        )
        self.encoder_proj_layers = nn.ModuleList(
            [nn.Linear(src_channels, tgt_channels) for _ in range(n_layers)]
        )
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=tgt_channels, num_heads=num_heads, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: torch.tensor of shape (batch, seq_len, tgt_channels)
            encoder_output: torch.tensor of shape (batch, seq_len, src_channels)

        returns:
            torch.tensor of shape (batch, seq_len, tgt_channels)
        """
        for self_attn_layer, encoder_projector, cross_attn_layer in zip(
            self.self_attn_layers, self.encoder_proj_layers, self.cross_attn_layers
        ):
            # b, s, tgt_channels
            self_attn_output = self_attn_layer(x)

            # b, s, src_channels -> b, s, tgt_channels
            projected_encoder_out = encoder_projector(encoder_output)

            # b, s, tgt_channels
            cross_attn_output, _ = cross_attn_layer(
                # b, s, tgt_channels
                self_attn_output,
                # b, s, tgt_channels
                projected_encoder_out,
                projected_encoder_out,
            )

            x = cross_attn_output

        return x


class Patcher(nn.Module):
    def __init__(self, in_seq_len: int, patch_size: int):
        """
        Args:
            in_seq_len: Length of the input sequence
            patch_size: Size of each patch (will be the stride of the convolution)
        """
        super().__init__()

        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(in_seq_len, in_seq_len // patch_size),
            nn.ReLU(),
            nn.Linear(in_seq_len // patch_size, in_seq_len // patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time, channels)
        Returns:
            Tensor of shape (batch, time//patch_size, channels)
        """
        # Convert from (batch, time, channels) to (batch, channels, time)
        x = rearrange(x, "b t c -> b c t")

        x = self.mlp(x)

        # Convert back to (batch, time//patch_size, channels)
        x = rearrange(x, "b c t -> b t c")

        return x


class Depatcher(nn.Module):
    def __init__(self, out_seq_len: int, patch_size: int):
        """
        Args:
            out_seq_len: Length of the output sequence
            patch_size: Size to expand each patch by (inverse of Patcher's patch_size)
        """
        super().__init__()

        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(out_seq_len // patch_size, out_seq_len),
            nn.ReLU(),
            nn.Linear(out_seq_len, out_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time//patch_size, channels)
        Returns:
            Tensor of shape (batch, time, channels)
        """
        # Convert from (batch, time//patch_size, channels) to (batch, channels, time//patch_size)
        x = rearrange(x, "b t c -> b c t")

        x = self.mlp(x)

        # Convert back to (batch, time, channels)
        x = rearrange(x, "b c t -> b t c")

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_length: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, seq_length, embed_dim))
        nn.init.uniform_(self.pos_embed, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed
