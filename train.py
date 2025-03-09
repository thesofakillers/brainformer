import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

from models.crossformer import CrossFormer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BrainFormer on synthetic data")

    # Data parameters
    parser.add_argument(
        "--seq_len", type=int, default=256, help="Sequence length for time series data"
    )
    parser.add_argument(
        "--input_channels", type=int, default=70, help="Number of input channels"
    )
    parser.add_argument(
        "--output_channels", type=int, default=300, help="Number of output channels"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/synthetic/processed",
        help="Directory to load data from",
    )
    parser.add_argument(
        "--split_data",
        action="store_true",
        default=True,
        help="Split raw data into train/val (if only inputs.pt/outputs.pt exist)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation data ratio when splitting raw data",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="inputs.pt",
        help="Filename of input data (default: inputs.pt)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs.pt",
        help="Filename of output data (default: outputs.pt)",
    )

    # Model parameters
    parser.add_argument("--num_heads", type=int, default=4, help="Num heads")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--num_enc_layers", type=int, default=4, help="Num enc layers")
    parser.add_argument("--num_dec_layers", type=int, default=4, help="Num dec layers")

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=7e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.3, help="Weight decay")
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.99, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        default=False,
        help="Whether to save checkpoints during training",
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log training metrics every N batches"
    )
    parser.add_argument(
        "--viz_every",
        type=int,
        default=1,
        help="Visualize sample predictions every N epochs",
    )

    # Optimizer parameters
    parser.add_argument(
        "--use_muon",
        action="store_true",
        default=False,
        help="Whether to use Muon optimizer for 2D parameters",
    )
    parser.add_argument(
        "--muon_momentum", type=float, default=0.95, help="Momentum for Muon optimizer"
    )
    parser.add_argument(
        "--muon_nesterov",
        action="store_true",
        default=True,
        help="Whether to use Nesterov momentum for Muon",
    )
    parser.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Number of Newton-Schulz iteration steps for Muon",
    )

    # Wandb parameters
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="brainformer", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb entity name"
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="gradients",
        help="Wandb watch mode: gradients, parameters, or all",
    )
    parser.add_argument(
        "--wandb_watch_log_freq",
        type=int,
        default=100,
        help="Frequency of logging gradients and parameters",
    )

    # Torch compile parameters
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Whether to use torch.compile for model acceleration",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        help="Torch compile mode: default, reduce-overhead, or max-autotune",
    )

    return parser.parse_args()


def prepare_data(args):
    """Load datasets for training and validation"""
    # Check if we need to split raw data
    inputs_path = os.path.join(args.data_dir, args.input_file)
    outputs_path = os.path.join(args.data_dir, args.output_file)

    train_data_path = os.path.join(args.data_dir, "train_inputs.pt")
    train_label_path = os.path.join(args.data_dir, "train_outputs.pt")
    val_data_path = os.path.join(args.data_dir, "val_inputs.pt")
    val_label_path = os.path.join(args.data_dir, "val_outputs.pt")

    # If split_data is True and raw files exist but split files don't, perform the split
    if (
        args.split_data
        and os.path.exists(inputs_path)
        and os.path.exists(outputs_path)
        and (not os.path.exists(train_data_path) or not os.path.exists(val_data_path))
    ):
        logger.info(
            f"Raw data found: {inputs_path} and {outputs_path}. Splitting into train and validation sets..."
        )

        # Load raw data
        inputs = torch.load(inputs_path)
        outputs = torch.load(outputs_path)

        # Determine split indices
        dataset_size = inputs.shape[0]
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        val_size = int(args.val_ratio * dataset_size)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        # Split the data
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        val_inputs = inputs[val_indices]
        val_outputs = outputs[val_indices]

        # Save the split data
        logger.info(f"Saving split data to {args.data_dir}")
        torch.save(train_inputs, train_data_path)
        torch.save(train_outputs, train_label_path)
        torch.save(val_inputs, val_data_path)
        torch.save(val_outputs, val_label_path)

        logger.info(
            f"Train data: inputs {train_inputs.shape}, outputs {train_outputs.shape}"
        )
        logger.info(
            f"Validation data: inputs {val_inputs.shape}, outputs {val_outputs.shape}"
        )
    else:
        missing_files = []
        for path in [train_data_path, train_label_path, val_data_path, val_label_path]:
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            # Check if raw data exists
            if os.path.exists(inputs_path) and os.path.exists(outputs_path):
                logger.info(
                    "Raw data found but split_data is False. Set --split_data to automatically split the data."
                )
            raise FileNotFoundError(
                f"Data files not found: {', '.join(missing_files)}. "
                f"Please ensure data exists in {args.data_dir} before training or use --split_data."
            )

        logger.info("Loading pre-split data...")
        train_inputs = torch.load(train_data_path)
        train_outputs = torch.load(train_label_path)
        val_inputs = torch.load(val_data_path)
        val_outputs = torch.load(val_label_path)

        logger.info(
            f"Train data: inputs {train_inputs.shape}, outputs {train_outputs.shape}"
        )
        logger.info(
            f"Validation data: inputs {val_inputs.shape}, outputs {val_outputs.shape}"
        )

    # Create datasets and dataloaders
    # Check data dimensions and ensure correct shape [batch, seq_len, channels]
    # Conv1DEncoder expects [batch, seq_len, channels] which it will transpose to [batch, channels, seq_len]
    logger.info("Checking data dimensions...")

    logger.info(
        f"Final train data shapes: inputs {train_inputs.shape}, outputs {train_outputs.shape}"
    )
    logger.info(
        f"Final validation data shapes: inputs {val_inputs.shape}, outputs {val_outputs.shape}"
    )

    train_dataset = TensorDataset(train_inputs, train_outputs)
    val_dataset = TensorDataset(val_inputs, val_outputs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader


def create_model(args):
    """Create a CrossFormer model with the specified configuration"""
    # Parse hidden dimension lists

    # Create model
    model = CrossFormer(
        src_num_channels=args.input_channels,  # EEG channels
        tgt_num_channels=args.output_channels,  # MEG channels
        max_sequence_length=args.seq_len,  # Sequence length
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
    )
    # Apply torch.compile if enabled
    if args.use_compile and hasattr(torch, "compile"):
        logger.info(f"Applying torch.compile with mode: {args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)
    elif args.use_compile and not hasattr(torch, "compile"):
        logger.warning(
            "torch.compile is not available in your PyTorch version. Continuing without compilation."
        )

    return model


def train_epoch(
    model, dataloader, criterion, optimizer, device, args, epoch, scheduler=None
):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs[:, : args.seq_len, :]
        targets = targets[:, : args.seq_len, :]

        # Forward pass with teacher forcing
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        if isinstance(optimizer, list):
            # If we have multiple optimizers (e.g., Muon + AdamW)
            for opt in optimizer:
                opt.zero_grad()
            loss.backward()
            for opt in optimizer:
                opt.step()
        else:
            # Single optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update metrics
        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Log batch metrics
        if batch_idx % args.log_every == 0:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}"
            )

        # Log batch metrics to wandb
        if args.use_wandb and batch_idx % args.log_every == 0:
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "batch": batch_idx + epoch * num_batches,
                    "epoch": epoch,
                }
            )
    # Step the scheduler if provided
    if scheduler is not None:
        scheduler.step()
        if args.use_wandb:
            wandb.log(
                {
                    "learning_rate": optimizer.param_groups[0]["lr"]
                    if not isinstance(optimizer, list)
                    else optimizer[1].param_groups[0]["lr"]
                }
            )

    return epoch_loss / num_batches


def validate(model, dataloader, criterion, device, seq_len):
    """Validate the model on the validation set"""
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs[:, :seq_len, :]
            targets = targets[:, :seq_len, :]

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update metrics
            val_loss += loss.item()

    # Calculate average validation loss
    val_loss /= len(dataloader)
    return val_loss


def save_checkpoint(model, optimizer, epoch, loss, args, is_best=False):
    """Save model checkpoint"""
    os.makedirs(args.save_dir, exist_ok=True)

    # Checkpoint filename
    filename = os.path.join(args.save_dir, f"brainformer_epoch_{epoch + 1}.pt")

    # Save checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "args": vars(args),
    }

    # Handle optimizer state
    if isinstance(optimizer, list):
        # If using multiple optimizers (e.g., Muon + AdamW)
        checkpoint["optimizer_state_dict"] = [opt.state_dict() for opt in optimizer]
    else:
        # Single optimizer
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, filename)

    logger.info(f"Checkpoint saved to {filename}")

    # Save best model if this is the best so far
    if is_best:
        best_filename = os.path.join(args.save_dir, "brainformer_best.pt")
        torch.save(checkpoint, best_filename)
        logger.info(f"Best model saved to {best_filename}")


def plot_training_history(train_losses, val_losses, args):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs(args.save_dir, exist_ok=True)
    plot_path = os.path.join(args.save_dir, "loss_history.png")
    plt.savefig(plot_path)
    logger.info(f"Loss history plot saved to {plot_path}")


def visualize_predictions(model, dataloader, device, epoch, args):
    """
    Visualize sample input, ground truth, and predicted output

    Args:
        model: The trained model
        dataloader: Data loader containing validation data
        device: The device to use for inference
        epoch: Current epoch number
        args: Command line arguments
    """
    model.eval()

    # Get a batch of data
    inputs, targets = next(iter(dataloader))

    # Select just one sample to visualize but keep batch dimension
    input_sample = inputs[:1].to(device)  # Shape: [1, seq_len, input_channels]
    target_sample = targets[:1].to(device)  # Shape: [1, seq_len, output_channels]

    # Ensure we're using the correct sequence length
    input_sample = input_sample[:, : args.seq_len, :]
    target_sample = target_sample[:, : args.seq_len, :]

    # Generate prediction
    with torch.no_grad():
        prediction = model(input_sample)

    # Move tensors to CPU for plotting
    input_sample = input_sample.cpu().numpy()[0]  # Shape: [seq_len, input_channels]
    target_sample = target_sample.cpu().numpy()[0]  # Shape: [seq_len, output_channels]
    prediction = prediction.cpu().numpy()[0]  # Shape: [seq_len, output_channels]

    # Create directory for visualization
    viz_dir = os.path.join(args.save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Create a multi-panel figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Plot input channels (reducing dimensionality if needed)
    axs[0].set_title("Input Signal")
    if input_sample.shape[1] > 10:
        # If too many channels, just plot first 10
        channels_to_plot = min(10, input_sample.shape[1])
        for i in range(channels_to_plot):
            axs[0].plot(input_sample[:, i], label=f"Ch {i + 1}")
        axs[0].set_xlabel("Time Steps")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
    else:
        im = axs[0].imshow(input_sample.T, aspect="auto", interpolation="none")
        axs[0].set_xlabel("Time Steps")
        axs[0].set_ylabel("Channel")
        plt.colorbar(im, ax=axs[0])

    # Plot target output - similar logic as input
    axs[1].set_title("Target Output")
    if target_sample.shape[1] > 10:
        # If too many channels, just plot first 10
        channels_to_plot = min(10, target_sample.shape[1])
        for i in range(channels_to_plot):
            axs[1].plot(target_sample[:, i], label=f"Ch {i + 1}")
        axs[1].set_xlabel("Time Steps")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend()
    else:
        im = axs[1].imshow(target_sample.T, aspect="auto", interpolation="none")
        axs[1].set_xlabel("Time Steps")
        axs[1].set_ylabel("Channel")
        plt.colorbar(im, ax=axs[1])

    # Plot predicted output - similar logic as input
    axs[2].set_title("Predicted Output")
    if prediction.shape[1] > 10:
        # If too many channels, just plot first 10
        channels_to_plot = min(10, prediction.shape[1])
        for i in range(channels_to_plot):
            axs[2].plot(prediction[:, i], label=f"Ch {i + 1}")
        axs[2].set_xlabel("Time Steps")
        axs[2].set_ylabel("Amplitude")
        axs[2].legend()
    else:
        im = axs[2].imshow(prediction.T, aspect="auto", interpolation="none")
        axs[2].set_xlabel("Time Steps")
        axs[2].set_ylabel("Channel")
        plt.colorbar(im, ax=axs[2])

    # Add overall title
    plt.suptitle(f"Sample Visualization - Epoch {epoch + 1}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the figure
    viz_path = os.path.join(viz_dir, f"visualization.png")
    plt.savefig(viz_path)
    plt.close()

    logger.info(f"Sample visualization saved to {viz_path}")

    # Log to wandb if enabled
    if args.use_wandb:
        # Log regular visualization
        wandb_log_dict = {f"visualization": wandb.Image(viz_path)}

        # Create heatmap for input data
        fig_input, ax_input = plt.subplots(figsize=(12, 8))
        im_input = ax_input.imshow(
            input_sample.T, aspect="auto", interpolation="none", cmap="viridis"
        )
        ax_input.set_title("Input Data Heatmap")
        ax_input.set_xlabel("Time Steps")
        ax_input.set_ylabel("Input Channels")
        plt.colorbar(im_input, ax=ax_input)
        plt.tight_layout()
        wandb_log_dict[f"heatmap_input"] = wandb.Image(fig_input)

        # Create heatmap for target data
        fig_target, ax_target = plt.subplots(figsize=(12, 8))
        im_target = ax_target.imshow(
            target_sample.T, aspect="auto", interpolation="none", cmap="viridis"
        )
        ax_target.set_title("Target Data Heatmap")
        ax_target.set_xlabel("Time Steps")
        ax_target.set_ylabel("Output Channels")
        plt.colorbar(im_target, ax=ax_target)
        plt.tight_layout()
        wandb_log_dict[f"heatmap_target"] = wandb.Image(fig_target)

        # Create heatmap for prediction data
        fig_pred, ax_pred = plt.subplots(figsize=(12, 8))
        im_pred = ax_pred.imshow(
            prediction.T, aspect="auto", interpolation="none", cmap="viridis"
        )
        ax_pred.set_title("Prediction Data Heatmap")
        ax_pred.set_xlabel("Time Steps")
        ax_pred.set_ylabel("Output Channels")
        plt.colorbar(im_pred, ax=ax_pred)
        plt.tight_layout()
        wandb_log_dict[f"heatmap_prediction"] = wandb.Image(fig_pred)

        # Create heatmap for prediction vs target difference
        fig_diff, ax_diff = plt.subplots(figsize=(12, 8))
        diff_data = target_sample - prediction
        im_diff = ax_diff.imshow(
            diff_data.T, aspect="auto", interpolation="none", cmap="RdBu_r"
        )
        ax_diff.set_title("Prediction Error Heatmap (Target - Prediction)")
        ax_diff.set_xlabel("Time Steps")
        ax_diff.set_ylabel("Output Channels")
        plt.colorbar(im_diff, ax=ax_diff)
        plt.tight_layout()
        wandb_log_dict[f"heatmap_error"] = wandb.Image(fig_diff)

        # Log all images to wandb
        wandb.log(wandb_log_dict)

        # Close all figures
        plt.close(fig_input)
        plt.close(fig_target)
        plt.close(fig_pred)
        plt.close(fig_diff)


def main():
    # Parse command-line arguments
    args = parse_args()

    # Initialize distributed environment if using Muon
    rank = 0
    world_size = 1
    if args.use_muon:
        # Check if we're in a distributed environment (via torchrun)
        if "LOCAL_RANK" in os.environ:
            # Initialize the distributed environment
            import torch.distributed as dist

            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if not dist.is_initialized():
                # Initialize the process group
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )

            print(
                f"Initialized distributed environment: rank={rank}, world_size={world_size}"
            )

    # Initialize wandb if enabled
    if args.use_wandb:
        # Only initialize wandb on rank 0 when using distributed training
        if not args.use_muon or rank == 0:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
            )

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Determine the device to use
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader = prepare_data(args)

    # Create model
    model = create_model(args)
    model = model.to(device)

    # Log model configuration
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Set up wandb to watch model gradients if wandb is enabled
    if args.use_wandb:
        wandb.watch(
            model,
            log=args.wandb_watch,
            log_freq=args.wandb_watch_log_freq,
            log_graph=True,
        )

    # Loss function and optimizer
    criterion = nn.MSELoss()

    # Configure optimizer using the model's method
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device,
        use_muon=args.use_muon,
        muon_momentum=args.muon_momentum,
        muon_nesterov=args.muon_nesterov,
        muon_ns_steps=args.muon_ns_steps,
        rank=rank,
        world_size=world_size,
    )

    # Setup learning rate scheduler with cosine annealing and linear warmup
    if isinstance(optimizer, list):
        # For Muon + AdamW case, we only apply scheduler to AdamW
        opt_for_scheduler = optimizer[1]
    else:
        opt_for_scheduler = optimizer

    # Linear warmup for 1 epoch
    warmup_scheduler = LinearLR(
        opt_for_scheduler,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=len(train_loader),
    )

    # Cosine annealing for remaining epochs
    cosine_scheduler = CosineAnnealingLR(
        opt_for_scheduler, T_max=(args.epochs - 1) * len(train_loader)
    )

    # Combine schedulers: linear warmup followed by cosine annealing
    scheduler = SequentialLR(
        opt_for_scheduler,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[len(train_loader)],  # Switch after 1 epoch
    )

    logger.info("Using cosine LR schedule with 1 epoch linear warmup")

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    logger.info("Starting training...")

    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, args, epoch, scheduler
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, args.seq_len)
        val_losses.append(val_loss)

        # Log epoch results
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Log to wandb if enabled
        if args.use_wandb:
            wandb_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1,
            }
            wandb.log(wandb_metrics)

        # Check if this is the best model so far
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        # Save checkpoint
        if args.save_checkpoints and (
            (epoch + 1) % args.save_every == 0 or is_best or epoch == args.epochs - 1
        ):
            save_checkpoint(model, optimizer, epoch, val_loss, args, is_best)

        # Visualize predictions
        if (epoch + 1) % args.viz_every == 0:
            visualize_predictions(model, val_loader, device, epoch, args)

    # Plot training history
    plot_training_history(train_losses, val_losses, args)

    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
