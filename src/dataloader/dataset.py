import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class MultimodalSpatiotemporalDataset(Dataset):
    """
    Multimodal Dataset for Cross-Modal Spatiotemporal Transportation Forecasting.
    
    Loads time-series state feeds (speed/volume/density), spatial graph structures, 
    and constructs textual prompt contexts for alignment with MLLM backbones.
    """
    def __init__(
        self,
        time_series_data: np.ndarray,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        history_steps: int = 12,
        pred_steps: int = 12,
        tokenizer: Optional[Any] = None,
        max_token_len: int = 128,
        text_metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            time_series_data (np.ndarray): Array of shape (Total_Timesteps, Num_Nodes, Features).
            edge_index (torch.Tensor): Graph edge connectivity matrix of shape (2, Num_Edges).
            edge_weight (torch.Tensor, optional): Edge weights/distances of shape (Num_Edges,).
            history_steps (int): Number of past observation time steps (default: 12 -> 1 hr @ 5min).
            pred_steps (int): Number of future target time steps to predict (default: 12 -> 1 hr).
            tokenizer (Any, optional): HuggingFace/MLLM tokenizer instance.
            max_token_len (int): Maximum sequence length for prompt tokenization.
            text_metadata (List[Dict], optional): Temporal/spatial metadata per timestep 
                                                  (e.g., timestamp, weather, incidents).
        """
        super().__init__()
        self.raw_data = torch.tensor(time_series_data, dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.history_steps = history_steps
        self.pred_steps = pred_steps
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.text_metadata = text_metadata

        self.num_samples = len(self.raw_data) - history_steps - pred_steps + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Data length ({len(self.raw_data)}) is too short for history_steps={history_steps} "
                f"and pred_steps={pred_steps}."
            )

    def __len__(self) -> int:
        return self.num_samples

    def _construct_text_prompt(self, idx: int) -> str:
        """
        Generates a natural language contextual prompt describing the spatiotemporal window.
        """
        if self.text_metadata and idx < len(self.text_metadata):
            meta = self.text_metadata[idx]
            timestamp = meta.get("timestamp", "Unknown time")
            day_of_week = meta.get("day_of_week", "Unknown day")
            weather = meta.get("weather", "Clear")
            incident = meta.get("incident", "No active traffic incidents reported.")
        else:
            timestamp = f"Timestep index {idx}"
            day_of_week = "Weekday"
            weather = "Normal"
            incident = "Routine traffic flow conditions."

        prompt = (
            f"<|im_start|>system\nYou are an expert urban traffic assistant. Predict future network states "
            f"based on historical graph dynamics and local contextual signals.<|im_end|>\n"
            f"<|im_start|>user\nTime: {timestamp} ({day_of_week}). Weather: {weather}. "
            f"Advisory: {incident}\nForecast task: Estimate road network states for the next {self.pred_steps} steps.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, PyGData, Dict[str, torch.Tensor]]]:
        """
        Returns:
            Dict containing:
                - 'x': Historical sensor tensor of shape (history_steps, Num_Nodes, Features)
                - 'y': Target sensor tensor of shape (pred_steps, Num_Nodes, Features)
                - 'graph': PyTorch Geometric Data object containing network topology
                - 'prompt_text': Raw string context prompt
                - 'tokenized_prompt': (Optional) Tokenized inputs dictionary if tokenizer is supplied
        """
        # Slice temporal windows
        x_start = idx
        x_end = idx + self.history_steps
        y_end = x_end + self.pred_steps

        x_tensor = self.raw_data[x_start:x_end]  # (T_in, N, F)
        y_tensor = self.raw_data[x_end:y_end]    # (T_out, N, F)

        # Build PyG Data object for graph operations
        pyg_graph = PyGData(
            x=x_tensor.transpose(0, 1),  # Reshape to (N, T_in, F) for node feature layout
            edge_index=self.edge_index,
            edge_attr=self.edge_weight
        )

        # Generate language prompt context
        raw_prompt = self._construct_text_prompt(x_end)

        sample = {
            "x": x_tensor,
            "y": y_tensor,
            "graph": pyg_graph,
            "prompt_text": raw_prompt
        }

        # Tokenize prompt if a tokenizer is attached
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                raw_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_token_len,
                return_tensors="pt"
            )
            sample["tokenized_prompt"] = {k: v.squeeze(0) for k, v in tokenized.items()}

        return sample


# --- Sanity Test Script ---
if __name__ == "__main__":
    print("Testing MultimodalSpatiotemporalDataset initialization...")
    
    # Mock data dimensions: 100 timesteps, 10 road nodes, 2 features (speed, volume)
    mock_series = np.random.randn(100, 10, 2).astype(np.float32)
    mock_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    mock_edge_weight = torch.rand(9, dtype=torch.float32)

    dataset = MultimodalSpatiotemporalDataset(
        time_series_data=mock_series,
        edge_index=mock_edge_index,
        edge_weight=mock_edge_weight,
        history_steps=12,
        pred_steps=12
    )

    sample = dataset[0]
    print(f"Dataset length: {len(dataset)}")
    print(f"Historical tensor shape 'x': {sample['x'].shape}")          # [12, 10, 2]
    print(f"Target tensor shape 'y':     {sample['y'].shape}")          # [12, 10, 2]
    print(f"PyG Graph Node features:     {sample['graph'].x.shape}")    # [10, 12, 2]
    print(f"Constructed Context Prompt:\n{sample['prompt_text']}")