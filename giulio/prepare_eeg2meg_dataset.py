import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from scipy import signal


def load_eeg2megset(cache_dir: Optional[str] = None, split: str = "train") -> Dict:
    """
    Load the EEG2MEG dataset from Hugging Face.

    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load ('train', 'validation', or 'test')

    Returns:
        Dict: The loaded dataset
    """
    print(f"Loading EEG2MEG dataset (split: {split})...")
    dataset = load_dataset(
        "fracapuano/eeg2meg-medium", split=split, cache_dir=cache_dir
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    return dataset


def process_dataset(
    dataset: Dict, expected_seq_len: int = 771
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process the EEG2MEG dataset and transpose data to sequence-first format.

    Args:
        dataset: The loaded dataset
        expected_seq_len: Expected sequence length

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input (EEG) and output (MEG) tensors
                                          in shape [batch, sequence_length, channels]
    """
    num_samples = len(dataset)

    # Create empty lists to store the processed data
    inputs = []
    outputs = []

    # Process each sample
    for i in tqdm(range(num_samples), desc="Processing dataset", ncols=100):
        # Get EEG data (input) and MEG data (output)
        eeg = dataset[i]["eeg"]
        meg = dataset[i]["meg"]

        # Convert to float tensors
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        meg_tensor = torch.tensor(meg, dtype=torch.float32)

        # Print information about the first few samples for debugging
        if i < 3:
            print(
                f"Sample {i} EEG shape: {eeg_tensor.shape}, dtype: {eeg_tensor.dtype}"
            )
            print(
                f"Sample {i} EEG min: {eeg_tensor.min()}, max: {eeg_tensor.max()}, mean: {eeg_tensor.mean()}"
            )
            print(
                f"Sample {i} MEG shape: {meg_tensor.shape}, dtype: {meg_tensor.dtype}"
            )
            print(
                f"Sample {i} MEG min: {meg_tensor.min()}, max: {meg_tensor.max()}, mean: {meg_tensor.mean()}"
            )

        # Ensure data is shaped as [channels, sequence_length]
        # If 1D, reshape to [1, sequence_length]
        if eeg_tensor.ndim == 1:
            eeg_tensor = eeg_tensor.unsqueeze(0)

        if meg_tensor.ndim == 1:
            meg_tensor = meg_tensor.unsqueeze(0)

        # Ensure the sequence length is correct
        if eeg_tensor.shape[1] != expected_seq_len:
            if eeg_tensor.shape[1] < expected_seq_len:
                # Pad to expected length
                padding = (0, expected_seq_len - eeg_tensor.shape[1])
                eeg_tensor = torch.nn.functional.pad(eeg_tensor, padding)
            else:
                # Truncate to expected length
                eeg_tensor = eeg_tensor[:, :expected_seq_len]

        if meg_tensor.shape[1] != expected_seq_len:
            if meg_tensor.shape[1] < expected_seq_len:
                # Pad to expected length
                padding = (0, expected_seq_len - meg_tensor.shape[1])
                meg_tensor = torch.nn.functional.pad(meg_tensor, padding)
            else:
                # Truncate to expected length
                meg_tensor = meg_tensor[:, :expected_seq_len]

        # Transpose both tensors from [channels, sequence_length] to [sequence_length, channels]
        eeg_tensor = eeg_tensor.transpose(0, 1)  # Now [sequence_length, channels]
        meg_tensor = meg_tensor.transpose(0, 1)  # Now [sequence_length, channels]

        inputs.append(eeg_tensor)
        outputs.append(meg_tensor)

    # Stack all samples into batch dimension
    inputs_tensor = torch.stack(inputs)  # [batch, sequence, channels]
    outputs_tensor = torch.stack(outputs)  # [batch, sequence, channels]

    print(
        f"Processed data shapes: Inputs {inputs_tensor.shape}, Outputs {outputs_tensor.shape}"
    )
    return inputs_tensor, outputs_tensor


def save_dataset(
    inputs: torch.Tensor, outputs: torch.Tensor, save_dir: str, prefix: str = "eeg2meg"
) -> None:
    """
    Save the processed dataset to files.

    Args:
        inputs: Input tensor (EEG data)
        outputs: Output tensor (MEG data)
        save_dir: Directory to save the dataset
        prefix: Prefix for the filenames
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save tensors with progress tracking
    with tqdm(total=2, desc="Saving tensors", ncols=100) as save_progress:
        torch.save(inputs, os.path.join(save_dir, f"{prefix}_inputs.pt"))
        save_progress.update(1)
        torch.save(outputs, os.path.join(save_dir, f"{prefix}_outputs.pt"))
        save_progress.update(1)

    print("Creating visualization sample...")
    # Save a sample plot for visualization
    sample_idx = 0  # Using the first sample for consistency

    # Create a comprehensive visualization
    plt.figure(figsize=(15, 15))

    # Plot input channels (EEG)
    plt.subplot(3, 1, 1)
    plt.title("Input Signal (EEG)", fontsize=14)
    num_input_channels_to_plot = min(10, inputs.shape[2])
    for c in range(num_input_channels_to_plot):
        plt.plot(inputs[sample_idx, :, c].numpy(), label=f"Ch {c + 1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True, alpha=0.3)

    # Plot output channels (MEG)
    plt.subplot(3, 1, 2)
    plt.title("Output Signal (MEG)", fontsize=14)
    num_output_channels_to_plot = min(10, outputs.shape[2])
    for c in range(num_output_channels_to_plot):
        plt.plot(outputs[sample_idx, :, c].numpy(), label=f"Ch {c + 1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True, alpha=0.3)

    # Add a spectrogram of the EEG input (first channel) to check frequency content
    plt.subplot(3, 1, 3)
    plt.title("EEG Channel 1 Spectrogram", fontsize=14)
    f, t, Sxx = signal.spectrogram(inputs[sample_idx, :, 0].numpy(), fs=100)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Power/Frequency (dB/Hz)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_sample_plot.png"))

    # Create a separate figure for power spectral density analysis
    plt.figure(figsize=(15, 10))

    # Plot power spectral density of input channels
    plt.subplot(2, 1, 1)
    plt.title("EEG Power Spectral Density", fontsize=14)
    for c in range(num_input_channels_to_plot):
        f, Pxx = signal.welch(inputs[sample_idx, c].numpy(), fs=100, nperseg=256)
        plt.semilogy(f, Pxx, label=f"Ch {c + 1}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power/Frequency [V^2/Hz]")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True, alpha=0.3)

    # Plot power spectral density of output channels
    plt.subplot(2, 1, 2)
    plt.title("MEG Power Spectral Density", fontsize=14)
    for c in range(num_output_channels_to_plot):
        f, Pxx = signal.welch(outputs[sample_idx, c].numpy(), fs=100, nperseg=256)
        plt.semilogy(f, Pxx, label=f"Ch {c + 1}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power/Frequency [V^2/Hz]")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_spectral_analysis.png"))

    # Save metadata
    with open(os.path.join(save_dir, f"{prefix}_metadata.txt"), "w") as f:
        f.write(f"Input (EEG) shape: {inputs.shape}\n")
        f.write(f"Output (MEG) shape: {outputs.shape}\n")
        f.write(f"Input type: {inputs.dtype}\n")
        f.write(f"Output type: {outputs.dtype}\n")
        f.write(
            f"Input stats - Min: {inputs.min().item()}, Max: {inputs.max().item()}, Mean: {inputs.mean().item()}, Std: {inputs.std().item()}\n"
        )
        f.write(
            f"Output stats - Min: {outputs.min().item()}, Max: {outputs.max().item()}, Mean: {outputs.mean().item()}, Std: {outputs.std().item()}\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare EEG2MEG dataset from Hugging Face"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./eeg2meg", help="Directory to save dataset"
    )
    parser.add_argument(
        "--prefix", type=str, default="eeg2meg", help="Prefix for saved files"
    )
    parser.add_argument(
        "--seq_len", type=int, default=351, help="Expected sequence length"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache the dataset"
    )

    args = parser.parse_args()

    # Create progress tracking for overall process
    with tqdm(
        total=3, desc="Overall progress", position=0, ncols=100
    ) as overall_progress:
        # Load the dataset
        dataset = load_eeg2megset(cache_dir=args.cache_dir, split=args.split)
        overall_progress.update(1)

        # Process the dataset
        print("Processing dataset...")
        inputs, outputs = process_dataset(dataset, expected_seq_len=args.seq_len)
        overall_progress.update(1)

        # Save the dataset
        print(f"Saving processed dataset to {args.save_dir}...")
        save_dataset(
            inputs=inputs, outputs=outputs, save_dir=args.save_dir, prefix=args.prefix
        )
        overall_progress.update(1)

        # Complete
        print("Dataset preparation complete!")
        print(f"Input (EEG) shape: {inputs.shape}")
        print(f"Output (MEG) shape: {outputs.shape}")


if __name__ == "__main__":
    main()
