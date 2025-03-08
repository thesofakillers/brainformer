import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from model.architectures.transformer import EncoderDecoderTransformer
from model.architectures.twostage_attention import TwoStageAttentionLayer


class EncoderDecoderCrossFormer(EncoderDecoderTransformer):
    """
    An Encoder Decoder CrossFormer model, performing two-stages attention on data in the shape (B, T, D)
    with B: batch size, T: sequence length, D: dimension of the time series.

    As per the problem setting, masking is exclusively defined to the time dimension.
    Also, the dimension of the time-series is considered to be permutation-invariant and is thus
    not considered in the positional encoding for the model.
    """

    def __init__(
        self,
        source_sequence_dimension: int,
        target_sequence_dimension: int,
        router_dim: int = 10,
        src_vocab_size: int = 255,
        tgt_vocab_size: int = 255,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_seq_length: int = 8192,
    ):
        # Cross former specific elements
        self.src_sequence_dimension = source_sequence_dimension
        self.tgt_sequence_dimension = target_sequence_dimension
        self.router_dim = router_dim

        self.source_channel_embedding = self._create_positional_encoding(
            source_sequence_dimension, embed_dim
        )
        self.target_channel_embedding = self._create_positional_encoding(
            target_sequence_dimension, embed_dim
        )

        super().__init__(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_seq_length=max_seq_length,
        )

        # cross-attend the encoder output while decoding
        self.decoder_cross_attention_encoder = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=self.encoder[
                        0
                    ].time_attention.dropout,  # same dropout throughout network
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def _stack_two_stage_attention_layers(
        self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                TwoStageAttentionLayer(
                    ts_length=self.max_sequence_length,
                    router_dim=self.router_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def _instantiate_encoder(
        self, embed_dim: int, num_heads: int, ff_dim: int, num_encoder_layers: int
    ) -> nn.ModuleList:
        """Instantiates the encoder module list as a stack of TwoStageAttentionLayers"""
        return self._stack_two_stage_attention_layers(
            embed_dim, num_heads, ff_dim, num_encoder_layers
        )

    def _instantiate_decoder(
        self, embed_dim: int, num_heads: int, ff_dim: int, num_decoder_layers: int
    ) -> nn.ModuleList:
        """Instantiates the decoder module list as a stack of TwoStageAttentionLayers"""
        return self._stack_two_stage_attention_layers(
            embed_dim, num_heads, ff_dim, num_decoder_layers
        )

    def forward(
        self,
        src,
        tgt,
        src_src_time_mask=None,
        src_src_dimension_mask=None,
        src_src_key_padding_time_mask=None,
        tgt_tgt_time_mask=None,
        tgt_tgt_dimension_mask=None,
        tgt_tgt_key_padding_time_mask=None,
        tgt_src_dimension_mask=None,
    ):
        """Overrides parent's forward method to perform two-stages attention on data in the shape (B, T, D)"""
        batch, src_seq_len, src_channels = src.shape
        batch, tgt_seq_len, tgt_channels = tgt.shape

        assert src_seq_len == tgt_seq_len, (
            "Source and Target sequence have different lenghts! Check padding."
        )

        # Create time-wise causal mask for the target sequence
        if tgt_tgt_time_mask is None:
            tgt_tgt_time_mask = self._create_causal_mask(tgt_seq_len).to(tgt.device)

        # Get the embeddings for the tokens in the sequence
        src = self.src_embedding(src)
        pos_embeddings = repeat(
            self.positional_encoding[:, : src.size(1)].to(src.device),
            "b t embed_dim -> b t ts_dim embed_dim",
            ts_dim=src_channels,
        )
        channel_embeddings = repeat(
            self.source_channel_embedding[:, : src.size(2)].to(src.device),
            "b src_ts_dim embed_dim -> b t src_ts_dim embed_dim",
            t=src_seq_len,
        )
        src += pos_embeddings + channel_embeddings
        # scaling down the input by model dimension
        src *= math.sqrt(src.size(-1))

        # pass the input sequence through the encoder
        encoder_output = src
        for encoder_layer in self.encoder:
            encoder_output = encoder_layer(
                encoder_output,
                attention_mask_time=src_src_time_mask,
                key_padding_mask_time=src_src_key_padding_time_mask,
                attention_mask_dimension=src_src_dimension_mask,
            )

        # Embed the tokens in the target sequence
        tgt = self.tgt_embedding(tgt)
        positional_embeddings = repeat(
            self.positional_encoding[:, : tgt.size(1)].to(tgt.device),
            "b t embed_dim -> b t ts_dim embed_dim",
            ts_dim=tgt_channels,
        )
        channel_embeddings = repeat(
            self.target_channel_embedding[:, : tgt.size(2)].to(tgt.device),
            "b tgt_ts_dim embed_dim -> b t tgt_ts_dim embed_dim",
            t=tgt_seq_len,
        )
        tgt += positional_embeddings + channel_embeddings
        tgt *= math.sqrt(tgt.size(-1))

        # cross-attending dimension wise requires reshaping the encoder output
        encoder_output = rearrange(
            encoder_output,
            "b T d_source embed_dim -> (b T) d_source embed_dim",
            b=batch,
            d_source=src_channels,
        )

        # Decoding pipeline, self-attending and cross-attending from encoder
        decoder_output = tgt
        for i, (decoder_layer, cross_attention_to_encoder) in enumerate(
            zip(self.decoder, self.decoder_cross_attention_encoder)
        ):
            # self attending the decoder input
            decoder_self_attention = decoder_layer(
                decoder_output,
                attention_mask_time=tgt_tgt_time_mask,
                key_padding_mask_time=tgt_tgt_key_padding_time_mask,
            )

            decoder_for_cross_attention = rearrange(
                decoder_self_attention,
                "b T d_target embed_dim -> (b T) d_target embed_dim",
                b=batch,
                T=tgt_seq_len,
            )

            # cross-attending the encoder output
            decoder_output, _ = cross_attention_to_encoder(
                decoder_for_cross_attention,
                encoder_output,
                encoder_output,
                attn_mask=tgt_src_dimension_mask,
            )

            decoder_output = rearrange(
                decoder_output,
                "(b T) d_target embed_dim -> b T d_target embed_dim",
                b=batch,
                T=tgt_seq_len,
            )

        return self.output_layer(decoder_output)

    def visualize(self, save_path="model_visualization.png"):
        """
        Creates a visual representation of the model's architecture using torchviz.

        Args:
            save_path (str): Path where the visualization will be saved

        Returns:
            None: Saves the visualization to the specified path
        """
        try:
            from torchviz import make_dot
        except ImportError:
            print("Please install torchviz: pip install torchviz")
            return

        # Create sample input
        batch_size = 1
        seq_len = 257
        src = torch.randint(0, 255, (batch_size, seq_len, self.src_sequence_dimension))
        tgt = torch.randint(0, 255, (batch_size, seq_len, self.tgt_sequence_dimension))

        # Get model output
        output = self(src, tgt)

        # Create visualization
        dot = make_dot(output, params=dict(self.named_parameters()))

        # Customize graph appearance
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr("node", shape="box")

        # Save visualization
        dot.render(save_path, format="png", cleanup=True)
        print(f"Model visualization saved to {save_path}")


if __name__ == "__main__":
    model = EncoderDecoderCrossFormer(
        source_sequence_dimension=10,
        target_sequence_dimension=10,
        router_dim=10,
        max_seq_length=4,
    )

    src = torch.randint(0, 255, (1, 4, 10))
    tgt = torch.randint(0, 255, (1, 4, 20))

    print(model(src, tgt).shape)

    # Add visualization example
    model.visualize()
