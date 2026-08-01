import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class SpatialGraphEncoder(nn.Module):
    """
    Spatiotemporal Graph Neural Network (ST-GNN) Encoder.
    Combines Graph Convolution / Graph Attention with Temporal Convolutions
    to extract spatiotemporal embeddings across network nodes.
    """
    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_heads: int = 4,
        gnn_type: str = "gat"
    ):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        
        # Spatial Graph Convolution Layer
        if self.gnn_type == "gat":
            self.spatial_conv = GATConv(
                in_channels=in_channels,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True
            )
        else:
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
        edge_weight: torch.Tensor = None
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

        # Step 1: Reshape tensor to apply Spatial GNN across all timesteps
        # (B * T_in * N, F)
        x_flat = x.reshape(B * T_in * N, F_dim)

        # Repeat graph edges for batched timesteps
        if self.gnn_type == "gat":
            h_spatial = self.spatial_conv(x_flat, edge_index)
        else:
            h_spatial = self.spatial_conv(x_flat, edge_index, edge_weight)

        h_spatial = F.relu(h_spatial)
        # Reshape to (B, T_in, N, hidden_dim)
        h_spatial = h_spatial.view(B, T_in, N, -1)

        # Step 2: Temporal Convolution across time step sequence
        # Permute to (B * N, hidden_dim, T_in) for Conv1d
        h_temp = h_spatial.permute(0, 2, 3, 1).reshape(B * N, -1, T_in)
        h_temp = self.temporal_conv(h_temp)  # (B * N, out_dim, T_in)

        # Max-pool across historical timesteps to capture global temporal signals
        h_pooled, _ = torch.max(h_temp, dim=-1)  # (B * N, out_dim)

        # Reshape back to (B, N, out_dim)
        out = h_pooled.view(B, N, -1)
        out = self.proj(out)

        return out


# --- Sanity Check Script ---
if __name__ == "__main__":
    print("Testing SpatialGraphEncoder...")
    encoder = SpatialGraphEncoder(in_channels=2, hidden_dim=64, out_dim=128)
    
    mock_x = torch.randn(32, 12, 10, 2)  # (Batch, T_in, Nodes, Features)
    mock_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    out = encoder(mock_x, mock_edge_index)
    print(f"Output embedding shape: {out.shape}")  # Expected: [32, 10, 128]