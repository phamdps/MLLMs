"""
METR-LA Dataset Loader for Multimodal Traffic Prediction
Handles loading of historical sensor readings, construction of sliding windows,
and formatting numerical sequences into text-promptable representations.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class METRLADataset(Dataset):
    def __init__(self, data_path: str, hist_steps: int = 12, pred_steps: int = 12, mode: str = "train"):
        """
        Args:
            data_path (str): Path to the METR-LA h5 or csv data file.
            hist_steps (int): Number of historical time steps (e.g., 12 steps = 1 hour).
            pred_steps (int): Number of prediction steps into the future.
            mode (str): 'train', 'val', or 'test'.
        """
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        self.mode = mode
        
        # Load raw sensor data (assuming standard METR-LA numpy or pandas structure)
        self.data = self._load_raw_data(data_path)
        self.samples = self._create_sliding_windows()

    def _load_raw_data(self, path: str) -> np.ndarray:
<<<<<<< HEAD
        if path.endswith(".h5"):
            df = pd.read_hdf(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path, index_col=0)
        else:
            # Fallback or synthetic simulation data for quick sandbox testing if file doesn't exist yet
            if not os.path.exists(path):
                print(f"⚠️ Warning: Path {path} not found. Generating synthetic sensor stream for pipeline validation.")
                return np.random.rand(5000, 207) # 5000 time steps, 207 sensors
            df = pd.read_csv(path)
        
=======
        # If a directory is passed by mistake, append the default filename
        if os.path.isdir(path):
            path = os.path.join(path, "metr_la.h5")
            
        if path.endswith(".h5") and os.path.exists(path):
            try:
                df = pd.read_hdf(path)
            except Exception:
                # Fallback if PyTables/h5 is missing or incompatible
                df = pd.read_csv(path.replace(".h5", ".csv")) if os.path.exists(path.replace(".h5", ".csv")) else None
        elif path.endswith(".csv") and os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
        else:
            df = None

        if df is None:
            print(f"⚠️ Warning: Valid METR-LA data file not found at {path}. Generating synthetic sensor stream for pipeline validation.")
            return np.random.rand(5000, 207)  # 5000 time steps, 207 sensors
            
>>>>>>> 13332e1 (add Agent)
        return df.values

    def _create_sliding_windows(self):
        num_samples = len(self.data) - self.hist_steps - self.pred_steps + 1
        
        # Simple data split: 70% train, 15% val, 15% test
        train_len = int(num_samples * 0.7)
        val_len = int(num_samples * 0.15)
        
        if self.mode == "train":
            start_idx, end_idx = 0, train_len
        elif self.mode == "val":
            start_idx, end_idx = train_len, train_len + val_len
        else:
            start_idx, end_idx = train_len + val_len, num_samples
            
        return [(i, i + self.hist_steps, i + self.hist_steps + self.pred_steps) 
                for i in range(start_idx, end_idx)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_h, end_h, end_p = self.samples[idx]
        
        history_seq = self.data[start_h:end_h]     # Shape: (hist_steps, num_sensors)
        target_seq = self.data[end_h:end_p]        # Shape: (pred_steps, num_sensors)
        
        return {
            "history": torch.tensor(history_seq, dtype=torch.float32),
            "target": torch.tensor(target_seq, dtype=torch.float32)
        }

def get_metr_la_dataloader(data_path: str, batch_size: int = 4, mode: str = "train"):
    dataset = METRLADataset(data_path=data_path, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))