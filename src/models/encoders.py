"""
Spatiotemporal Graph Neural Network (ST-GNN) Encoder using GCNConv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class SpatialGraphEncoder(nn.Module):
    """
    Spatiotemporal Graph Neural Network (ST-GNN) Encoder.
    Uses GCNConv for robust, stable spatial message passing on METR-LA.
    """
    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_heads: int = 4,
        gnn_type: str = "gcn"
    ):
        super().__init__()
        # Force GCNConv to prevent any accidental GAT sparse index assertion crashes
        self.spatial_conv = GCNConv(in_channels, hidden_dim)

        # Temporal Feature Encoder (1D Convolution along time axis)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=out_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Final projection layer to unify node features
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node feature tensor of shape (B, T_in, N, F)
            edge_index: Graph edge indices tensor of shape (2, E)
            edge_weight: Graph edge weights tensor of shape (E,)

        Returns:
            torch.Tensor: Spatiotemporal node representations of shape (B, N, out_dim)
        """
        B, T_in, N, F_dim = x.shape
        device = x.device

        # Ensure edge tensors reside on the correct device
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # --- SAFETY CHECK FOR OUT-OF-BOUNDS EDGE INDICES ---
        max_node_idx = edge_index.max().item()
        if max_node_idx >= N:
            # Option A: Filter out invalid edges automatically to prevent crashes
            valid_mask = (edge_index[0] < N) & (edge_index[1] < N)
            edge_index = edge_index[:, valid_mask]
            if edge_weight is not None:
                edge_weight = edge_weight[valid_mask]
        # --------------------------------------------------

        # Apply spatial graph convolution across batch and time dimensions safely
        spatial_outputs = []
        for b_idx in range(B):
            batch_t_outputs = []
            for t_idx in range(T_in):
                node_feat = x[b_idx, t_idx]  # Shape: (N, F)
                h = self.spatial_conv(node_feat, edge_index, edge_weight=edge_weight)
                batch_t_outputs.append(h)
            
            spatial_outputs.append(torch.stack(batch_t_outputs, dim=0))
            
        h_spatial = torch.stack(spatial_outputs, dim=0)  # Shape: (B, T_in, N, hidden_dim)
        h_spatial = F.relu(h_spatial)

        # Temporal Convolution across time step sequence
        h_temp = h_spatial.permute(0, 2, 3, 1).reshape(B * N, -1, T_in)
        h_temp = self.temporal_conv(h_temp)  # Shape: (B * N, out_dim, T_in)

        h_pooled, _ = torch.max(h_temp, dim=-1)  # Shape: (B * N, out_dim)

        out = h_pooled.view(B, N, -1)
        out = self.proj(out)

        return out