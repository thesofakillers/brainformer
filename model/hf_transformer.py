from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from model.architectures.transformer import EncoderDecoderTransformer
from model.architectures.crossformer import EncoderDecoderCrossFormer
from model.hf_configs import Seq2SeqConfig, Seq2SeqCrossConfig
from einops import rearrange
from tqdm import tqdm

class Seq2SeqTransformer(PreTrainedModel):
    """
    Custom Transformer for Sequence to Sequence tasks.
    """
    config_class = Seq2SeqConfig
    base_model_prefix = "transformer"
    
    def __init__(self, config: PretrainedConfig, device: Optional[str]=None):
        super().__init__(config)
        self.config = config
        self.softmax = nn.Softmax(dim=-1)

        self.transformer = EncoderDecoderTransformer(
            src_vocab_size=config.vocab_size_src,
            tgt_vocab_size=config.vocab_size_tgt,
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            ff_dim=config.d_ff,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            max_seq_length=config.sequence_length
        )

    def _init_weights(self, module):
        """Initialize weights using He (Kaiming) initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')

    def _create_padding_mask(self, ids: torch.LongTensor) -> torch.DoubleTensor:
        """Creates a mask to avoid padded tokens to be interfering with attention"""
        # First create boolean mask where True = padding token
        is_padding = ids.eq(self.config.pad_token_id)
        
        # Convert to float and replace padding positions with -inf, others with 1.0
        mask = is_padding.float()
        mask = mask.masked_fill(is_padding, float('-inf'))
        mask = mask.masked_fill(~is_padding, 1.0)
        return mask

    def _shift_right(self, x: torch.LongTensor) -> torch.LongTensor:
        """Helper method to prepare decoder inputs (teacher forcing) by shifting right label tokens"""
        shifted = torch.full(
            (*x.shape[:-1], 1), 
            self.config.bos_token_id, 
            dtype=x.dtype, 
            device=x.device
        )
        shifted = torch.cat([shifted, x[:, :-1]], dim=-1)
        return shifted
    
    def _add_beginning_of_stream(self, x: torch.LongTensor) -> torch.LongTensor:
        """
        Helper method to add BOS token to the beginning of input sequences
        """
        bos = torch.full(
            (*x.shape[:-1], 1),
            self.config.bos_token_id,
            dtype=x.dtype,
            device=x.device
        )

        return torch.cat([bos, x], dim=-1)

    def _add_end_of_stream(self, x: torch.LongTensor) -> torch.LongTensor:
        """Helper method to add EOS token to the end of label sequences"""
        eos = torch.full(
            (*x.shape[:-1], 1),
            self.config.eos_token_id,
            dtype=x.dtype,
            device=x.device
        )
        return torch.cat([x, eos], dim=-1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs
    ) -> Union[Tuple, dict]:
        # TODO: add/end of streaming and right shift should take place outside of the model in tokenizer
        
        # adding beginning of stream tokens to input too
        input_ids = self._add_beginning_of_stream(input_ids)

        # adding end of stream tokens to labels
        labels = self._add_end_of_stream(labels)
        # Prepare input for the decoder
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        src_key_padding_mask = self._create_padding_mask(input_ids)
        tgt_key_padding_mask = self._create_padding_mask(decoder_input_ids)
        
        # Forward pass through your model
        outputs = self.transformer(
            src=input_ids,
            tgt=decoder_input_ids,
            src_mask=attention_mask,
            tgt_mask=decoder_attention_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(outputs.view(-1, self.config.vocab_size_tgt), labels.view(-1))

        return dict(
            loss=loss,
            logits=outputs,
        )
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> torch.LongTensor:
                
        batch_size = input_ids.shape[0]
        max_length = max_length or self.config.max_length or 128
        
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        
        for _ in range(max_length - 1):
            outputs = self.forward(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            
            next_token_logits = outputs["logits"][:, -1, :]
            
            if do_sample:
                # Apply temperature scaling
                scaled_logits = next_token_logits / temperature
                # Convert to probabilities
                next_token_probs = self.softmax(scaled_logits)
                # Sample from the probability distribution
                next_token = torch.multinomial(
                    next_token_probs, num_samples=1
                ).squeeze(-1)
            else:
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1)
            
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token.unsqueeze(-1)],
                dim=-1
            )
            
            # Stop if all sequences have generated EOS token
            if (decoder_input_ids == self.config.eos_token_id).any(dim=-1).all():
                break
                
        return decoder_input_ids


class Seq2SeqCrossFormer(Seq2SeqTransformer):
    """CrossFormer wrapper predicting over a discrete vocabulatory."""
    config_class = Seq2SeqCrossConfig
    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.softmax = nn.Softmax(dim=-1)

        self.transformer = EncoderDecoderCrossFormer(
            source_sequence_dimension=config.source_sequence_dimension,
            target_sequence_dimension=config.target_sequence_dimension,
            router_dim=config.router_dim,
            src_vocab_size=config.vocab_size_src,
            tgt_vocab_size=config.vocab_size_tgt,
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            ff_dim=config.d_ff,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            max_seq_length=config.sequence_length
        )

    def _shift_right(self, x: torch.LongTensor) -> torch.LongTensor:
        """
        Helper method to prepare decoder inputs (teacher forcing) by shifting right label tokens.
        Handles 3D (B, S, C) tensors
        """
        # Create shape that matches x's dimensions except for seq_len which will be 1
        shape = list(x.shape)
        shape[-2] = 1  # Set sequence dimension to 1
        
        shifted = torch.full(
            shape,
            self.config.bos_token_id,
            dtype=x.dtype,
            device=x.device
        )
        shifted = torch.cat([shifted, x[..., :-1, :]], dim=-2)
        return shifted
    
    def _add_beginning_of_stream(self, x: torch.LongTensor) -> torch.LongTensor:
        """
        Helper method to add BOS token to the beginning of input sequences.
        Handles 3D (B, S, C) tensors
        """
        shape = list(x.shape)
        shape[-2] = 1  # Set sequence dimension to 1
        sos = torch.full(
            shape, 
            self.config.bos_token_id, 
            dtype=x.dtype, 
            device=x.device
        )

        return torch.cat([sos, x], dim=-2)

    def _add_end_of_stream(self, x: torch.LongTensor) -> torch.LongTensor:
        """
        Helper method to add EOS token to the end of label sequences.
        Handles 3D (B, S, C) tensors
        """
        # Create shape that matches x's dimensions except for seq_len which will be 1
        shape = list(x.shape)
        shape[-2] = 1  # Set sequence dimension to 1
        
        eos = torch.full(
            shape,
            self.config.eos_token_id,
            dtype=x.dtype,
            device=x.device
        )
        return torch.cat([x, eos], dim=-2)

    def forward(
            self, 
            input_ids: torch.LongTensor,
            labels: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            **kwargs
            ):
        # FIXME: add/end of streaming and right shift should take place outside of the model in tokenizer

        # (in tokenizer) adding beginning of stream tokens to input too
        input_ids = self._add_beginning_of_stream(input_ids)

        # (in tokenizer) adding end of stream tokens to labels
        if labels is not None:
            labels = self._add_end_of_stream(labels)

        # Prepare input for the decoder
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        src_src_key_padding_time_mask = rearrange(
            self._create_padding_mask(
                input_ids
            ),
            'b s c -> (b c) s'
        )

        tgt_tgt_key_padding_time_mask = rearrange(
            self._create_padding_mask(
                decoder_input_ids
            ),
            'b s c -> (b c) s'
        )
        
        # Forward pass through your model
        outputs = self.transformer(
            src=input_ids,
            tgt=decoder_input_ids,
            src_src_time_mask=kwargs.get("src_src_time_mask"),
            src_src_dimension_mask=kwargs.get("src_src_dimension_mask"),
            src_src_key_padding_time_mask=src_src_key_padding_time_mask,
            tgt_tgt_time_mask=kwargs.get("tgt_tgt_time_mask"),
            tgt_tgt_dimension_mask=kwargs.get("tgt_tgt_dimension_mask"),
            tgt_tgt_key_padding_time_mask=tgt_tgt_key_padding_time_mask,
            tgt_src_dimension_mask=kwargs.get("tgt_src_dimension_mask")
        )
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                outputs.view(-1, self.config.vocab_size_tgt), labels.view(-1)
            )

        return dict(
            loss=loss,
            logits=outputs,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor]=None,
        max_length: Optional[int]=None,
        temperature: float=1.0,
        do_sample: bool=False,
        **kwargs
    ) -> torch.LongTensor:

        batch_size, timesteps, channels = input_ids.shape

        tgt_sequence_timesteps = timesteps+1  # src will gain a token at forward
        max_length = max_length or self.config.max_length or 128

        decoder_input_ids = torch.full(
            (
                batch_size, 
                tgt_sequence_timesteps,
                self.config.target_sequence_dimension
            ),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device
        )

        # Set BOS token at the start
        decoder_input_ids[:, 0, :] = self.config.bos_token_id
        generation_logits = []

        generation_length = timesteps # FIXME: when forecasting, use timesteps+max_length
        for t in tqdm(range(generation_length), desc="Generating sequence"):
            outputs = self.forward(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions for this timestep
            next_token_logits = outputs["logits"][:, t, :]
            generation_logits.append(
                next_token_logits.squeeze().detach().cpu().numpy()
            )

            if do_sample:
                scaled_logits = next_token_logits / temperature
                next_token_probs = self.softmax(scaled_logits)
                next_token = torch.multinomial(
                    next_token_probs, num_samples=1
                ).squeeze(-1)
            else:
                next_token = next_token_logits.argmax(dim=-1)

            try:
                # Place the predicted token at position t+1
                decoder_input_ids[:, t+1, :] = next_token
            except IndexError:
                break  # last token for fixed-size element
            
            # Check if all sequences have generated EOS token
            if (next_token == self.config.eos_token_id).all():
                pass  # deployed model, breaks here
            
            # For forecasting (which is not trained! Use with caution), we only keep the last tgt_sequence_timesteps
            decoder_input_ids = decoder_input_ids[
                :, -tgt_sequence_timesteps:, :
            ]
        return decoder_input_ids, generation_logits
