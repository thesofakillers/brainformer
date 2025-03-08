from einops import rearrange
import torch
from torch import nn
from twostage_attention import TwoStageAttentionLayer


class CrossFormer(nn.module):
    def __init__(
        self,
        source_num_channels: int,
        target_num_channels: int,
        max_sequence_length: int,
        num_heads: int,
        num_enc_layers: int,
        num_dec_layers: int,
    ) -> None:
        super().__init__()

        self.source_num_channels = source_num_channels
        self.target_num_channels = target_num_channels
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        self.encoder = EncoderBlock(
            self.max_sequence_length,
            self.source_num_channels,
            self.num_heads,
            self.num_enc_layers,
        )
        self.decoder = DecoderBlock(
            self.max_sequence_length,
            self.target_num_channels,
            self.num_heads,
            self.num_dec_layers,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(src)

        decoder_output = self.decoder(tgt, encoder_output)

        return decoder_output

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """Create a causal mask for the decoder. Prevents decoder from attending to future tokens."""
        mask = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
        return mask.to(dtype=torch.float)


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
        for encoder in self.self_attn_layers:
            x = encoder(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, seq_len, num_channels, num_heads, n_layers):
        super().__init__()
        self.self_attn_layers = nn.ModuleList(
            [
                TwoStageAttentionLayer(seq_len, num_channels, num_heads)
                for _ in range(n_layers)
            ]
        )
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=num_channels, num_heads=num_heads, batch_first=True
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
        for self_attn_layer, cross_attn_layer in zip(
            self.self_attn_layers, self.cross_attn_layers
        ):
            # b, s, tgt_channels
            self_attn_output = self_attn_layer(x)

            # b, s, tgt_channels
            cross_attn_output = cross_attn_layer(
                # b, s, tgt_channels
                self_attn_output,
                # b, s, src_channels
                encoder_output,
                encoder_output,
            )

        return cross_attn_output
