"""Minimal script to register a model running a custom architecture config and push it to the hub."""

from model.hf_configs import Seq2SeqCrossConfig
from model.hf_transformer import Seq2SeqCrossFormer

if __name__ == "__main__":
    Seq2SeqCrossFormer.register_for_auto_class()
    Seq2SeqCrossFormer.register_for_auto_class("AutoModel")

    config = Seq2SeqCrossConfig()
    model = Seq2SeqCrossFormer(config)

    model.push_to_hub("fracapuano/bwaves")

    del model

    """Then use the model pushed on the hub with the following"""
    model = Seq2SeqCrossFormer.from_pretrained(
        "fracapuano/bwaves", trust_remote_code=True
    )

    print(model)
