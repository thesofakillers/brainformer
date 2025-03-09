import torch
from torch import nn

from models.components import EncoderBlock, DecoderBlock, Patcher, Depatcher


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
