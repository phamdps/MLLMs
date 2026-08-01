import pickle
import numpy as np
import pandas as pd
import h5py
import torch
from typing import Tuple, Dict, Any
from src.dataloader.dataset import MultimodalSpatiotemporalDataset


def load_h5_data(h5_filename: str) -> np.ndarray:
    """
    Robust HDF5 loader using h5py to avoid Python 3.11 bytes/string 
    compatibility issues with pandas.read_hdf on legacy METR-LA files.
    """
    try:
        # First attempt: Try reading via h5py
        with h5py.File(h5_filename, 'r') as f:
            # Inspection: find key inside file (e.g., 'df', 'df/block0_values', etc.)
            keys = list(f.keys())
            if 'df' in keys:
                group = f['df']
                if 'block0_values' in group:
                    data = group['block0_values'][:]
                elif 'axis0' in group:
                    data = np.array(group['block0_values'])
                else:
                    data = group[:]
            else:
                # Fallback to the first available key
                first_key = keys[0]
                data = f[first_key][:]
            return data
    except Exception as e:
        print(f"h5py direct read fallback triggered due to: {e}")
        # Secondary fallback: standard pandas
        df = pd.read_hdf(h5_filename)
        return df.values


def load_pickle_matrix(pkl_filename: str) -> np.ndarray:
    """Loads the spatial distance/adjacency matrix from a pickle file."""
    with open(pkl_filename, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except UnicodeDecodeError:
            pickle_data = pickle.load(f)
    
    if isinstance(pickle_data, (list, tuple)):
        adj_mx = pickle_data[2]
    else:
        adj_mx = pickle_data
    return adj_mx


def get_edge_index_and_weights(adj_mx: np.ndarray, threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts an adjacency matrix into PyG edge_index and edge_weight tensors."""
    adj_mx[adj_mx < threshold] = 0.0
    edges = np.where(adj_mx > 0)
    
    edge_index = torch.tensor(np.array(edges), dtype=torch.long)
    edge_weight = torch.tensor(adj_mx[edges], dtype=torch.float32)
    
    return edge_index, edge_weight


def prepare_metr_la_datasets(
    h5_path: str,
    pkl_path: str,
    history_steps: int = 12,
    pred_steps: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1
) -> Dict[str, Any]:
    """
    Full robust loader for real METR-LA dataset.
    """
    # 1. Load traffic speed array safely
    raw_array = load_h5_data(h5_path)
    
    # Ensure shape is (Time_Steps, Num_Sensors, Features)
    if raw_array.ndim == 2:
        raw_array = raw_array[..., np.newaxis]

    # Standardize data based on training split metrics
    num_samples = len(raw_array)
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    mean = np.mean(raw_array[:train_end])
    std = np.std(raw_array[:train_end])
    normalized_data = (raw_array - mean) / std

    # 2. Load Adjacency Graph Topology
    adj_mx = load_pickle_matrix(pkl_path)
    edge_index, edge_weight = get_edge_index_and_weights(adj_mx)

    # 3. Slice splits
    train_data = normalized_data[:train_end]
    val_data = normalized_data[train_end:val_end]
    test_data = normalized_data[val_end:]

    # 4. Instantiate Datasets
    datasets = {
        "train": MultimodalSpatiotemporalDataset(
            time_series_data=train_data,
            edge_index=edge_index,
            edge_weight=edge_weight,
            history_steps=history_steps,
            pred_steps=pred_steps
        ),
        "val": MultimodalSpatiotemporalDataset(
            time_series_data=val_data,
            edge_index=edge_index,
            edge_weight=edge_weight,
            history_steps=history_steps,
            pred_steps=pred_steps
        ),
        "test": MultimodalSpatiotemporalDataset(
            time_series_data=test_data,
            edge_index=edge_index,
            edge_weight=edge_weight,
            history_steps=history_steps,
            pred_steps=pred_steps
        ),
        "scaler": {"mean": mean, "std": std}
    }
    
    return datasets


if __name__ == "__main__":
    print("Testing robust METR-LA data loader...")
    h5_file = "data/raw/METR-LA/metr_la.h5"
    pkl_file = "data/raw/METR-LA/adj_mx.pkl"
    
    data_dict = prepare_metr_la_datasets(h5_file, pkl_file)
    train_ds = data_dict["train"]
    sample = train_ds[0]
    
    print("\n✅ Successfully loaded real dataset with h5py!")
    print(f"Train samples count:      {len(train_ds)}")
    print(f"Input tensor x shape:     {sample['x'].shape}")
    print(f"Target tensor y shape:    {sample['y'].shape}")
    print(f"Graph Nodes count:        {sample['graph'].x.shape[0]}")