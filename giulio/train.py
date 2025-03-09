import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb
from crossformer import CrossFormer
from tqdm import tqdm
import argparse
from dataset import EEGMEGDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CrossFormer model")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--project_name", type=str, default="crossformer", help="WandB project name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("--num_heads", type=int, default=4, help="Num heads")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--num_enc_layers", type=int, default=4, help="Num enc layers")
    parser.add_argument("--num_dec_layers", type=int, default=4, help="Num dec layers")

    return parser.parse_args()


def train_crossformer(
    model: CrossFormer,
    train_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
    project_name: str = "crossformer-giulio",
):
    """
    Training loop for CrossFormer model

    Args:
        model: CrossFormer model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        project_name: WandB project name
    """
    # Initialize wandb
    wandb.init(project=project_name)
    wandb.watch(model, log="all", log_freq=100)

    # Setup training
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for src, tgt, target in progress_bar:
            src = src.to(device)
            tgt = tgt.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            progress_bar.set_postfix({"train_loss": loss.item()})

        avg_train_loss = train_loss / train_batches

        # Log metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
            }
        )

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Average Train Loss: {avg_train_loss:.6f}")


if __name__ == "__main__":
    args = parse_args()

    # Initialize model
    model = CrossFormer(
        src_num_channels=70,  # EEG channels
        tgt_num_channels=306,  # MEG channels
        max_sequence_length=256,  # Sequence length
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
    )

    # Initialize datasets and dataloaders
    train_dataset = EEGMEGDataset(split="train", sequence_length=256)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Train the model with parsed arguments
    train_crossformer(
        model=model,
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        project_name=args.project_name,
    )
