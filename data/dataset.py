import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb


class ECGDataset(Dataset):
    def __init__(self, record_ids, window_size=512, noise_std=0.05):
        """
        Args:
            record_ids (list): List of MIT-BIH record numbers (e.g., ['100', '101'])
            window_size (int): Length of ECG segment
            noise_std (float): Standard deviation of Gaussian noise
        """
        self.window_size = window_size
        self.noise_std = noise_std
        self.segments = []

        for record_id in record_ids:
            record = wfdb.rdrecord(record_id, pn_dir='mitdb')
            signal = record.p_signal[:, 0]  # Use first ECG channel
            signal = self.normalize(signal)

            # Segment into fixed windows
            for i in range(0, len(signal) - window_size, window_size):
                segment = signal[i:i + window_size]
                self.segments.append(segment)

        self.segments = np.array(self.segments)

    def normalize(self, x):
        return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        clean_signal = self.segments[idx]
        clean_signal = torch.tensor(clean_signal, dtype=torch.float32)

        # Add Gaussian noise
        noise = torch.randn_like(clean_signal) * self.noise_std
        noisy_signal = clean_signal + noise

        # Add channel dimension: (1, 512)
        clean_signal = clean_signal.unsqueeze(0)
        noisy_signal = noisy_signal.unsqueeze(0)

        return noisy_signal, clean_signal
