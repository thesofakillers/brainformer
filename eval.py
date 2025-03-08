import torch
from datasets import load_dataset, Dataset
from model.hf_transformer import Seq2SeqCrossFormer
from model.hf_configs import Seq2SeqCrossConfig
from train import preprocess_dataset

model = Seq2SeqCrossFormer.from_pretrained("fracapuano/eeg2meg-small")

# from the model config
sequence_length = 256

dataset = (
    load_dataset("fracapuano/eeg2meg-medium-tokenized", split="train")
    .with_format("pt")
    .select(range(2))
)

maxlen_dataset = dataset.map(
    lambda x: preprocess_dataset(x, sequence_length),
    batched=True,
    remove_columns=dataset.column_names,
)

meg_generated = model.generate(
    maxlen_dataset["input_ids"][0].unsqueeze(0), max_length=0
)

labels = dataset["labels"][0].unsqueeze(0)

with open("meg_generated.pt", "wb") as f:
    torch.save(meg_generated, f)

with open("labels.pt", "wb") as f:
    torch.save(labels, f)
