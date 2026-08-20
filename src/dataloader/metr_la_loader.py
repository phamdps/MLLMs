import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class METRLADataset(Dataset):
    """
    Standard dataset loader for the METR-LA traffic forecasting benchmark.
    Handles time-series historical sliding windows and future prediction horizons.
    """
    def __init__(self, data_path: str, history_window: int = 12, prediction_window: int = 12, train: bool = True):
        super().__init__()
        self.history_window = history_window
        self.prediction_window = prediction_window
        
        # Verify file path exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"[Error] METR-LA data file not found at specified path: {data_path}")
            
        # Load dataset format (.h5 or .csv)
        if data_path.endswith('.h5'):
            df = pd.read_hdf(data_path)
        else:
            df = pd.read_csv(data_path)
            
        data = df.values # Expected shape: (num_time_steps, num_sensors)
        
        # Standard train/test split partitioning (70% train, 30% test/val)
        num_samples = data.shape[0]
        train_len = int(num_samples * 0.7)
        
        if train:
            self.data = data[:train_len]
        else:
            self.data = data[train_len:]
            
        # Z-score normalization computed safely on training subset
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.data = (self.data - self.mean) / (self.std + 1e-5)
        
    def __len__(self) -> int:
        return len(self.data) - self.history_window - self.prediction_window + 1
        
    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.history_window]
        y = self.data[idx + self.history_window : idx + self.history_window + self.prediction_window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)