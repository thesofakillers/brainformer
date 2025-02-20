import os
import torch
import argparse
from datasets import load_dataset
from datasets import Dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from model.hf_transformer import (
    Seq2SeqCrossFormer, 
    Seq2SeqCrossConfig
)
from model.hf_callbacks import (
    VisualizeGeneratedMEGCallback
)

import wandb
from transformers.integrations import WandbCallback

# logging gradients and parameters
os.environ["WANDB_WATCH"] = "all"


def preprocess_dataset(examples: dict, max_length: int) -> dict:
    """
    Preprocesses a batch of the dataset performing max_length splitting and padding.
    """
    eeg, meg = examples['eeg'].squeeze(), examples['meg'].squeeze()
    padded_data = []
    for data in [eeg, meg]:
        maxlength_data = []
        chunks = torch.split(data, max_length, dim=-1)  # splitting the timesteps

        maxlength_data.extend([c for c in chunks[:-1]])

        # padding the last chunk to max_length
        padding_length = max_length - chunks[-1].size(-1)
        maxlength_data.append(
            torch.nn.functional.pad(chunks[-1], (0, padding_length), value=0)
        )

        padded_data.append(maxlength_data)
    
    padded_eeg, padded_meg = padded_data
    return {
        'input_ids': torch.vstack(padded_eeg).transpose(2, 1).long(),
        'labels': torch.vstack(padded_meg).transpose(2, 1).long(),
    }


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Train EEG2MEG model')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of crossformer layers')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--sequence_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--num_train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--dummy-data', action='store_true', help='Run training on much smaller dummy dataset')
    parser.add_argument('--initialize_weights', action='store_false', help='Initialize weights from scratch')
    args = parser.parse_args()

    # Model config now uses CLI args
    config = Seq2SeqCrossConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
    )
    if not args.dummy_data:
        dataset = load_dataset(
            "fracapuano/eeg2meg-medium-tokenized", 
            split="train"
        ).with_format("pt")
        
        maxlen_dataset = dataset.map(
                lambda x: preprocess_dataset(x, args.sequence_length),
                batched=True,
                remove_columns=dataset.column_names
            ).train_test_split(test_size=0.1)

        train, eval = maxlen_dataset["train"], maxlen_dataset["test"]
    else:
        train = Dataset.from_dict(
            {
                "input_ids": [3+torch.randint(0, 256, (args.sequence_length, 70)).to(torch.int8) for _ in range(1)],
                "labels": [3+torch.randint(0, 256, (args.sequence_length, 306)).to(torch.int8) for _ in range(1)]
            }
        ).with_format("pt")
        eval = train

    # Initialize model
    model = Seq2SeqCrossFormer(config)
    print(model)

    if args.initialize_weights:
        print("Initializing model weights for training!")
        model.apply(model._init_weights)
    
    # Print number of parameters
    model_parameters = sum(p.numel() for p in model.parameters())
    config.model_size = model_parameters
    print(f"Number of parameters (M): {model_parameters/1e6:.2f}M")

    # Initialize wandb before trainer
    run = wandb.init(
        project="eeg2meg-debug",
        config=config.to_dict()
    )

    # Calculate steps per epoch and eval frequency
    total_steps = max(1, len(train)//args.batch_size)
    print("1 epoch = {} steps".format(total_steps))
    eval_freq = 5
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"experiments/{run.name}",
        eval_strategy="steps",
        eval_steps=eval_freq*total_steps,
        learning_rate=3e-4,
        warmup_ratio=0.2,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2*args.batch_size,
        num_train_epochs=args.num_train_epochs,
        report_to="wandb",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=eval_freq*total_steps,
        bf16=True,
        torch_compile=False,
        push_to_hub=False,
        metric_for_best_model="eval_loss"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        callbacks=[
            WandbCallback(),
            EarlyStoppingCallback(early_stopping_patience=5),
            VisualizeGeneratedMEGCallback(
                model=model,
                eval_dataset=eval
            ),
            
        ]
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()