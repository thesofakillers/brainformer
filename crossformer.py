from einops import rearrange
import torch
from torch import nn
from twostage_attention import TwoStageAttentionLayer


class CrossFormer(nn.Module):
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
            self.src_num_channels,
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

        self.depatcher = Depatcher(
            in_channels=self.tgt_num_channels, patch_size=self.patch_size
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_patches = self.src_patcher(src)
        encoder_output = self.encoder(src_patches)

        tgt_patches = self.tgt_patcher(tgt)
        decoder_output = self.decoder(tgt_patches, encoder_output)

        output = self.depatcher(decoder_output)

        return output


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
        for encoder in self.layers:
            x = encoder(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, seq_len, src_channels, tgt_channels, num_heads, n_layers):
        super().__init__()
        self.self_attn_layers = nn.ModuleList(
            [
                TwoStageAttentionLayer(seq_len, tgt_channels, num_heads)
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
        x = rearrange(x, "b t c -> b c t")

        # Apply convolution to get (batch, channels, time//patch_size)
        x = self.conv(x)

        # Convert back to (batch, time//patch_size, channels)
        x = rearrange(x, "b c t -> b t c")

        return x


class Depatcher(nn.Module):
    def __init__(self, in_channels: int, patch_size: int):
        """
        Args:
            in_channels: Number of input channels
            patch_size: Size of each patch (will be the stride of the transposed convolution)
        """
        super().__init__()

        self.patch_size = patch_size
        self.conv_transpose = nn.ConvTranspose1d(
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
            x: Input tensor of shape (batch, time//patch_size, channels)
        Returns:
            Tensor of shape (batch, time, channels)
        """
        # Convert from (batch, time//patch_size, channels) to (batch, channels, time//patch_size)
        x = rearrange(x, "b t c -> b c t")

        # Apply transposed convolution to get (batch, channels, time)
        x = self.conv_transpose(x)

        # Convert back to (batch, time, channels)
        x = rearrange(x, "b c t -> b t c")

        return x
