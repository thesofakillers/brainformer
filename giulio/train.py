import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb
from crossformer import CrossFormer
from tqdm import tqdm
import argparse
from dataset import EEGMEGDataset
import os
from pathlib import Path


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
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
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
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )

    return parser.parse_args()


def evaluate(
    model: CrossFormer, val_loader: DataLoader, criterion: nn.Module, device: str
):
    """
    Evaluate the model on the validation set.

    Args:
        model: CrossFormer model
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Device to evaluate on

    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for src, tgt, target in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            target = target.to(device)

            output = model(src, tgt)
            loss = criterion(output, target)

            val_loss += loss.item()
            val_batches += 1

    return val_loss / val_batches


def train_crossformer(
    model: CrossFormer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 3e-4,
    device: str = "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
    project_name: str = "crossformer-giulio",
    patience: int = 10,
):
    """
    Training loop for CrossFormer model

    Args:
        model: CrossFormer model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        project_name: WandB project name
        patience: Number of epochs to wait for improvement before early stopping
    """
    # Initialize wandb
    wandb.init(project=project_name)
    wandb.watch(model, log="all", log_freq=100)

    # Setup training
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize tracking variables
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

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
            global_step += 1

            # Log metrics for each batch
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                },
                step=global_step,
            )

            progress_bar.set_postfix({"train_loss": loss.item()})

        # Compute average training loss
        avg_train_loss = train_loss / train_batches

        # Validation phase
        avg_val_loss = evaluate(model, val_loader, criterion, device)

        # Log metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "global_step": global_step,
            },
            step=global_step,
        )

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Average Train Loss: {avg_train_loss:.6f}")
        print(f"Average Validation Loss: {avg_val_loss:.6f}")

        # Check if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save the best model to wandb
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }

            # Save checkpoint to a temporary file
            tmp_path = Path("model.pt")
            torch.save(checkpoint, tmp_path)

            # Log the model to wandb
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description=f"Model checkpoint from epoch {epoch} with val_loss={avg_val_loss:.6f}",
            )
            artifact.add_file(tmp_path)
            wandb.log_artifact(artifact)

            # Clean up temporary file
            tmp_path.unlink()

            print(f"New best model saved with validation loss: {avg_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


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
    val_dataset = EEGMEGDataset(split="validation", sequence_length=256)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=4,
        pin_memory=True,
    )

    # Train the model with parsed arguments
    train_crossformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        project_name=args.project_name,
        patience=args.patience,
    )
