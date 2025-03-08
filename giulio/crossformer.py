import torch
from torch import nn


class CrossFormer(nn.module):
    def __init__(
        self,
        source_num_channels: int,
        target_num_channels: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        self.source_num_channels = source_num_channels
        self.target_num_channels = target_num_channels
        self.max_sequence_length = max_sequence_length

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pass
