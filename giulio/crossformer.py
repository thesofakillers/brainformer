from einops import rearrange
import torch
from torch import nn
from twostage_attention import TwoStageAttentionLayer


class CrossFormer(nn.module):
    def __init__(
        self,
        src_num_channels: int,
        tgt_num_channels: int,
        max_sequence_length: int,
        patch_size: int,
        num_heads: int,
        num_enc_layers: int,
        num_dec_layers: int,
    ) -> None:
        super().__init__()

        self.src_num_channels = src_num_channels
        self.tgt_num_channels = tgt_num_channels
        self.max_sequence_length = max_sequence_length
        self.coarsened_length = max_sequence_length // patch_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        self.encoder = EncoderBlock(
            self.coarsened_length,
            self.src_num_channels,
            self.num_heads,
            self.num_enc_layers,
        )
        self.decoder = DecoderBlock(
            self.coarsened_length,
            self.tgt_num_channels,
            self.num_heads,
            self.num_dec_layers,
        )

        self.src_patcher = Patcher(
            in_channels=src_num_channels, patch_size=self.patch_size
        )
        self.tgt_patcher = Patcher(
            in_channels=tgt_num_channels, patch_size=self.patch_size
        )

        self.depatcher = nn.Linear(self.coarse_length, self.max_sequence_length)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_patches = self.src_patcher(src)
        encoder_output = self.encoder(src_patches)

        tgt_patches = self.tgt_patcher(tgt)
        decoder_output = self.decoder(tgt_patches, encoder_output)

        output = self.depatcher(decoder_output)

        return output

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


class Patcher(nn.Module):
    def __init__(self, in_channels: int, patch_size: int):
        """
        Args:
            in_channels: Number of input channels
            patch_size: Size of each patch (will be the stride of the convolution)
        """
        super().__init__()

        self.patch_size = patch_size
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            groups=in_channels,  # Separate conv for each channel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time, channels)
        Returns:
            Tensor of shape (batch, time//patch_size, channels)
        """
        # Convert from (batch, time, channels) to (batch, channels, time)
        x = x.transpose(1, 2)

        # Apply convolution to get (batch, channels, time//patch_size)
        x = self.conv(x)

        # Convert back to (batch, time//patch_size, channels)
        x = x.transpose(1, 2)

        return x
