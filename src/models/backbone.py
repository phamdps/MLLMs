import torch
import torch.nn as nn
from typing import Dict, Optional
from src.models.encoders import SpatialGraphEncoder


class CrossModalProjector(nn.Module):
    """
    Connects graph encoder embeddings to the LLM token embedding space.
    """
    def __init__(self, encoder_dim: int = 128, llm_dim: int = 3584):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, encoder_dim)
        Returns:
            Tensor of shape (B, N, llm_dim) aligned with text embeddings
        """
        return self.projector(x)


class MultimodalTransportationMLLM(nn.Module):
    """
    Production MLLM Backbone for Cross-Modal Spatiotemporal Prediction.
    Unifies Spatial Graph Neural Encoders, Cross-Modal Connectors, and Multi-Task Decoders.
    """
    def __init__(
        self,
        in_channels: int = 2,
        graph_embed_dim: int = 128,
        num_nodes: int = 207,  # Default standard for METR-LA
        pred_steps: int = 12,
        llm_hidden_size: int = 3584  # Aligned with Qwen2-VL-7B-Instruct
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_steps = pred_steps
        self.llm_hidden_size = llm_hidden_size

        # 1. Spatial-Temporal Graph Encoder
        self.graph_encoder = SpatialGraphEncoder(
            in_channels=in_channels,
            hidden_dim=64,
            out_dim=graph_embed_dim,
            gnn_type="gat"
        )

        # 2. Cross-Modal Projector (Connector)
        self.projector = CrossModalProjector(
            encoder_dim=graph_embed_dim,
            llm_dim=self.llm_hidden_size
        )

        # 3. Multi-Task Prediction Decoders (Meso-flow & Macro-demand)
        self.meso_flow_head = nn.Sequential(
            nn.Linear(self.llm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, pred_steps * in_channels)
        )

        self.macro_demand_head = nn.Sequential(
            nn.Linear(self.llm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, pred_steps)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal joint forecasting.

        Args:
            x: Sensor time-series tensor (B, T_in, N, F)
            edge_index: Graph adjacency topology (2, E)
            edge_weight: Optional edge weights (E,)

        Returns:
            Dict containing predicted 'meso_flow', 'macro_demand', and 'multimodal_tokens'.
        """
        B, T_in, N, F_dim = x.shape

        # Step 1: Extract spatial-temporal graph embeddings via ST-GNN
        graph_feats = self.graph_encoder(x, edge_index, edge_weight)  # (B, N, graph_embed_dim)

        # Step 2: Project spatial embeddings into LLM cross-modal space
        multimodal_tokens = self.projector(graph_feats)  # (B, N, llm_hidden_size)

        # Step 3: Mean pooling across nodes for macro-level context representation
        macro_tokens = torch.mean(multimodal_tokens, dim=1)  # (B, llm_hidden_size)

        # Step 4: Multi-Task Predictions
        meso_out = self.meso_flow_head(multimodal_tokens)  # (B, N, T_out * F)
        meso_out = meso_out.view(B, N, self.pred_steps, F_dim).permute(0, 2, 1, 3)

        macro_out = self.macro_demand_head(macro_tokens)  # (B, T_out)

        return {
            "meso_flow": meso_out,
            "macro_demand": macro_out,
            "multimodal_tokens": multimodal_tokens
        }


# --- Sanity Check Script ---
if __name__ == "__main__":
    print("Testing MultimodalTransportationMLLM and Encoder integration...")
    
    model = MultimodalTransportationMLLM(
        in_channels=2,
        graph_embed_dim=128,
        num_nodes=10,
        pred_steps=12
    )

    mock_x = torch.randn(4, 12, 10, 2)  # (Batch=4, T_in=12, Nodes=10, Feats=2)
    mock_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    mock_edge_weight = torch.tensor([1.0, 0.8, 0.5, 0.9], dtype=torch.float32)

    outputs = model(mock_x, mock_edge_index, mock_edge_weight)
    print(f"Meso Traffic Flow shape:   {outputs['meso_flow'].shape}")        # [4, 12, 10, 2]
    print(f"Macro Travel Demand shape: {outputs['macro_demand'].shape}")     # [4, 12]
    print(f"Multimodal Token Space:    {outputs['multimodal_tokens'].shape}") # [4, 10, 3584]