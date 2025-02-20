"""A encoder-decoder transformer model to convert EEG to MEG."""

import math
import torch
import torch.nn as nn

from typing import Optional

from model.architectures.encoder import EncoderBlock
from model.architectures.decoder import DecoderBlock
from huggingface_hub import PyTorchModelHubMixin

class EncoderDecoderTransformer(nn.Module, PyTorchModelHubMixin):
    """
    A transformer model with an encoder and a decoder.
    Implements multi-dimensional timeseries forecasting using a channel-independent strategy.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int=512,
        num_heads: int=8,
        ff_dim: int=2048,
        num_encoder_layers: int=6,
        num_decoder_layers: int=6,
        max_seq_length: int=8192  # ~8 seconds at 1.1 kHz
    ):
        super().__init__()
        # both encoder and decoder share the same max sequence length
        self.max_sequence_length = max_seq_length
    
        self.encoder = self._instantiate_encoder(
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_encoder_layers
        )
        
        self.decoder = self._instantiate_decoder(
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_decoder_layers
        )
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.positional_encoding = self._create_positional_encoding(
            max_seq_length+1,  # sequences pick up a BOS token 
            embed_dim
        )

        # Output layer
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)

    def _instantiate_encoder(self, embed_dim, num_heads, ff_dim, num_encoder_layers):
        return EncoderBlock(
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_encoder_layers
        )

    def _instantiate_decoder(self, embed_dim, num_heads, ff_dim, num_decoder_layers):
        return DecoderBlock(
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_decoder_layers
        )

    def _create_positional_encoding(self, max_seq_length, embed_dim):
        """Create positional encoding for the input sequence."""
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pos_encoding = torch.zeros(max_seq_length, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """Create a causal mask for the decoder. Prevents decoder from attending to future tokens."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask.to(dtype=torch.float)

    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor]=None,
            src_key_padding_mask: Optional[torch.Tensor]=None,
            tgt_mask: Optional[torch.Tensor]=None,
            tgt_key_padding_mask: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        assert src.size(1) == self.max_sequence_length + 1, ( 
            "Source sequence shape is different from max sequence length. Check padding."
        )
        """Both source and target sequence have shape (B, T_source, C_source), (B, T_target, C_target)"""
        # TODO: add reshaping and "translation" layer to allow C_source -> C_target for channel independent communication

        # Create masks
        tgt_mask = self._create_causal_mask(tgt.size(1)).to(tgt.device)
        
        # Embed and add positional encoding
        src = self.src_embedding(src) 
        tgt = self.tgt_embedding(tgt)
        
        src = src * math.sqrt(src.size(-1)) + self.positional_encoding[:, :src.size(1)].to(src.device)
        tgt = tgt * math.sqrt(src.size(-1)) + self.positional_encoding[:, :tgt.size(1)].to(tgt.device)
        
        # Encoder
        enc_output = src
        for enc_layer in self.encoder.layers:
            enc_output = enc_layer(
                enc_output, 
                attention_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
        
        # Decoder
        dec_output = tgt
        for dec_layer in self.decoder.layers:
            dec_output = dec_layer(
                x=dec_output,
                cross_key=enc_output, 
                cross_value=enc_output,
                # decoder layers self-attend target sequence masks
                self_attention_mask=tgt_mask,
                self_padding_mask=tgt_key_padding_mask,
                # decoder layers cross-attend encoder sequence masks
                cross_attention_mask=src_mask,
                cross_padding_mask=src_key_padding_mask
            )
        
        # Output projection, on dim=-1 i.e. target vocab size
        return self.output_layer(dec_output)

    
if __name__ == "__main__":
    vocab_size = 10
    transformer = EncoderDecoderTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        embed_dim=16,
        num_heads=2,
        ff_dim=128,
        max_seq_length=6
    )
    # tensors have shape (batch_size, sequence_length)
    x = torch.randint(0, vocab_size, (5, 6))
    y = torch.randint(0, vocab_size, (5, 3))

    print(transformer(x, y).shape)

