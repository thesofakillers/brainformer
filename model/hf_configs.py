from transformers import PretrainedConfig


class Seq2SeqConfig(PretrainedConfig):
    """
    Custom Transformer Config.
    """

    model_type = "custom_code"

    def __init__(
        self,
        vocab_size_src=512,
        vocab_size_tgt=512,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sequence_length=8192,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

        # on both encoder and decoder, same sequence length
        self.sequence_length = sequence_length


class Seq2SeqCrossConfig(Seq2SeqConfig):
    """
    Subclasses Seq2SeqConfig and adds source and target sequence dimensions, for CrossFormer.
    """

    def __init__(
        self,
        source_sequence_dimension=70,
        target_sequence_dimension=306,
        router_dim=10,
        vocab_size_src=258,  # 255 levels + 2 for bos and eos + 1 padding
        vocab_size_tgt=258,
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=512,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sequence_length=64,
        **kwargs,
    ):
        super().__init__(
            vocab_size_src=vocab_size_src,
            vocab_size_tgt=vocab_size_tgt,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sequence_length=sequence_length,
            **kwargs,
        )
        self.source_sequence_dimension = source_sequence_dimension
        self.target_sequence_dimension = target_sequence_dimension
        self.router_dim = router_dim
