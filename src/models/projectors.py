"""
Modality Alignment Projectors
Projects non-textual feature tokens (e.g., historical sensor vector streams, GNN embeddings)
into the shared semantic token space of Qwen2-VL.
"""

import torch
import torch.nn as nn

class SensorToTextSpaceProjector(nn.Module):
    def __init__(self, input_dim: int = 207, hidden_dim: int = 512, output_dim: int = 4096):
        """
        Args:
            input_dim (int): Number of sensors or feature dimension (e.g., 207 for METR-LA).
            hidden_dim (int): Intermediate bottleneck dimension.
            output_dim (int): Hidden dimension of Qwen2-VL-7B (typically 4096).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, sensor_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_feats (torch.Tensor): Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            torch.Tensor: Projected tensor of shape (batch_size, seq_len, output_dim)
        """
        return self.net(sensor_feats)