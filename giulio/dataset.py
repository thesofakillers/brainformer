import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np


class EEGMEGDataset(Dataset):
    """Dataset class for EEG to MEG data."""

    def __init__(
        self, split="train", sequence_length=256, val_size=0.2, random_seed=42
    ):
        """
        Initialize the dataset.

        Args:
            split (str): Dataset split to use ('train' or 'validation')
            sequence_length (int): Length to truncate sequences to
            val_size (float): Fraction of data to use for validation (0.0 to 1.0)
            random_seed (int): Random seed for reproducibility
        """
        # Load the full dataset
        full_dataset = load_dataset("fracapuano/eeg2meg-medium", split="train")

        # Calculate split sizes
        total_size = len(full_dataset)
        val_samples = int(total_size * val_size)
        train_samples = total_size - val_samples

        # Create train/val splits
        if split == "train":
            self.dataset = full_dataset.select(range(train_samples))
        elif split == "validation":
            self.dataset = full_dataset.select(range(train_samples, total_size))
        else:
            raise ValueError(f"Invalid split {split}. Must be 'train' or 'validation'")

        self.sequence_length = sequence_length
        self.split = split

        # Initialize normalizers
        self.eeg_normalizer = None
        self.meg_normalizer = None

        # For validation split, we need to get normalization params from training data
        if split == "validation":
            # Temporarily load training data to compute normalization params
            train_data = full_dataset.select(range(train_samples))
            self._compute_normalization_params(train_data)
        else:
            self._compute_normalization_params(self.dataset)

    def _compute_normalization_params(self, data):
        """
        Compute mean and std for normalization.

        Args:
            data: Dataset to compute normalization parameters from
        """
        # Initialize arrays to store statistics
        eeg_means = []
        eeg_stds = []
        meg_means = []
        meg_stds = []

        # Compute statistics for each sample
        for sample in data:
            eeg_data = np.array(sample["eeg_data"])[
                :, : self.sequence_length
            ]  # (70, 256)
            meg_data = np.array(sample["meg_data"])[
                :, : self.sequence_length
            ]  # (306, 256)

            # Compute statistics along time dimension
            eeg_means.append(np.mean(eeg_data, axis=1))
            eeg_stds.append(np.std(eeg_data, axis=1))
            meg_means.append(np.mean(meg_data, axis=1))
            meg_stds.append(np.std(meg_data, axis=1))

        # Compute global statistics
        self.eeg_mean = np.mean(eeg_means, axis=0)  # (70,)
        self.eeg_std = np.mean(eeg_stds, axis=0)  # (70,)
        self.meg_mean = np.mean(meg_means, axis=0)  # (306,)
        self.meg_std = np.mean(meg_stds, axis=0)  # (306,)

        # Ensure no division by zero
        self.eeg_std = np.where(self.eeg_std == 0, 1e-6, self.eeg_std)
        self.meg_std = np.where(self.meg_std == 0, 1e-6, self.meg_std)

    def normalize_eeg(self, eeg_data):
        """Normalize EEG data."""
        return (eeg_data - self.eeg_mean[:, None]) / self.eeg_std[:, None]

    def normalize_meg(self, meg_data):
        """Normalize MEG data."""
        return (meg_data - self.meg_mean[:, None]) / self.meg_std[:, None]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            tuple: (eeg_data, meg_data) where:
                - eeg_data is source data of shape (sequence_length, 70)
                - meg_data is target data of shape (sequence_length, 306)
        """
        sample = self.dataset[idx]

        # Get EEG and MEG data and truncate to sequence_length
        eeg_data = np.array(sample["eeg_data"])[:, : self.sequence_length]  # (70, 256)
        meg_data = np.array(sample["meg_data"])[:, : self.sequence_length]  # (306, 256)

        # Normalize data
        eeg_data = self.normalize_eeg(eeg_data)
        meg_data = self.normalize_meg(meg_data)

        # Convert to torch tensors and ensure correct shape
        eeg_data = torch.FloatTensor(eeg_data).T  # (256, 70)
        meg_data = torch.FloatTensor(meg_data).T  # (256, 306)

        return eeg_data, meg_data, meg_data  # src, tgt, target
