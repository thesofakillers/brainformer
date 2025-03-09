import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

from models.components import EncoderBlock, DecoderBlock, Patcher, Depatcher, PositionalEmbedding


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

        # Initialize learnable starting sequence for decoder
        self.initial_tgt = nn.Parameter(
            torch.empty(1, max_sequence_length, tgt_num_channels)
        )
        nn.init.uniform_(self.initial_tgt, -0.01, 0.01)

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
            in_seq_len=max_sequence_length, patch_size=self.patch_size
        )
        self.tgt_patcher = Patcher(
            in_seq_len=max_sequence_length, patch_size=self.patch_size
        )
        self.pos_embed_encoder = PositionalEmbedding(self.coarsened_length, self.src_num_channels)
        self.pos_embed_decoder = PositionalEmbedding(self.coarsened_length, self.tgt_num_channels)

        self.depatcher = Depatcher(
            out_seq_len=self.max_sequence_length, patch_size=self.patch_size
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = src.shape
        # b, 256, C_src -> b, 64, C_src
        src_patches = self.src_patcher(src)
        src_patches = self.pos_embed_encoder(src_patches)
        encoder_output = self.encoder(src_patches)

        tgt = repeat(self.initial_tgt, "1 t c -> b t c", b=batch_size)
        # b, 256, C_tgt -> b, 64, C_tgt
        tgt_patches = self.tgt_patcher(tgt)
        tgt_patches = self.pos_embed_decoder(tgt_patches)
        decoder_output = self.decoder(tgt_patches, encoder_output)

        # b, 64, C -> b, 256, C
        output = self.depatcher(decoder_output)

        # constrain output to [-1, 1]
        output = F.tanh(output)

        return output

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
        use_muon=False,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        rank=0,
        world_size=1,
    ):
        """
        Configure optimizers for the entire model
        """
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Create optimizer groups
        # Weight decay for 2D parameters (weights), no decay for 1D parameters (biases, LayerNorms)
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        # Print parameter stats
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # If Muon is requested, use it for 2D parameters (typically weights) and AdamW for the rest
        if use_muon:
            from muon import Muon

            # Create Muon optimizer for 2D parameters
            muon_optimizer = Muon(
                decay_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
                rank=rank,  # Use the provided rank
                world_size=world_size,  # Use the provided world_size
            )
            print(
                f"Using Muon optimizer for {len(decay_params)} 2D parameter tensors (rank={rank}, world_size={world_size})"
            )

            # Create AdamW optimizer for 1D parameters
            # Use fused AdamW if available
            fused_available = (
                "fused" in torch.__dict__
                and hasattr(torch.optim, "AdamW")
                and hasattr(torch.optim.AdamW.__init__.__code__, "co_varnames")
                and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
            )
            use_fused = fused_available and device_type == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()

            adamw_optimizer = torch.optim.AdamW(
                [{"params": nodecay_params, "weight_decay": 0.0}],
                lr=learning_rate,
                betas=betas,
                **extra_args,
            )
            print(
                f"Using AdamW optimizer for {len(nodecay_params)} non-2D parameter tensors"
            )
            print(f"using fused AdamW: {use_fused}")

            # Return a list of optimizers (for use with PyTorch's ZeroRedundancyOptimizer)
            return [muon_optimizer, adamw_optimizer]
        else:
            # Standard optimization with AdamW
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]

            # Use fused AdamW if available
            fused_available = (
                "fused" in torch.__dict__
                and hasattr(torch.optim, "AdamW")
                and hasattr(torch.optim.AdamW.__init__.__code__, "co_varnames")
                and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
            )
            use_fused = fused_available and device_type == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()

            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, **extra_args
            )
            print(f"using fused AdamW: {use_fused}")

            return optimizer
