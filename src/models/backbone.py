"""
Core MLLM Fusion Backbone for Transportation Digital Twin
Combines Spatial-Temporal GNN graph embeddings, visual/textual features, 
and multi-task prediction decoders aligned with Qwen2-VL.
"""

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
    Unifies Spatial Graph Neural Encoders, Cross-Modal Connectors, Multi-Task Decoders,
    and optional Visual/Textual fusion channels.
    """
    def __init__(
        self,
        in_channels: int = 2,
        graph_embed_dim: int = 128,
        num_nodes: int = 207,  # Default standard for METR-LA
        pred_steps: int = 12,
        llm_hidden_size: int = 3584,  # Aligned with Qwen2-VL-7B-Instruct
        use_vision_tokens: bool = True
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_steps = pred_steps
        self.llm_hidden_size = llm_hidden_size
        self.use_vision_tokens = use_vision_tokens

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

        # 3. Optional Visual Modality Adapter (for CCTV video features)
        if self.use_vision_tokens:
            self.vision_adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(3, self.llm_hidden_size),  # Assuming RGB channels or feature maps
                nn.ReLU()
            )

        # 4. Multi-Task Prediction Decoders (Meso-flow & Macro-demand)
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
        edge_weight: Optional[torch.Tensor] = None,
        vision_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal joint forecasting.

        Args:
            x: Sensor time-series tensor (B, T_in, N, F)
            edge_index: Graph adjacency topology (2, E)
            edge_weight: Optional edge weights (E,)
            vision_tensor: Optional CCTV video frame tensor (B, T_v, C, H, W)

        Returns:
            Dict containing predicted 'meso_flow', 'macro_demand', and 'multimodal_tokens'.
        """
        B, T_in, N, F_dim = x.shape

        # Step 1: Extract spatial-temporal graph embeddings via ST-GNN
        graph_feats = self.graph_encoder(x, edge_index, edge_weight)  # (B, N, graph_embed_dim)

        # Step 2: Project spatial embeddings into LLM cross-modal space
        multimodal_tokens = self.projector(graph_feats)  # (B, N, llm_hidden_size)

        # Step 3: Optional Vision Modality Fusion
        if self.use_vision_tokens and vision_tensor is not None:
            # vision_tensor shape: (B, T_v, C, H, W) -> Flatten temporal & batch for adapter
            Tv = vision_tensor.shape[1]
            v_flat = vision_tensor.view(-1, *vision_tensor.shape[2:]) # (B * Tv, C, H, W)
            v_embeds = self.vision_adapter(v_flat).view(B, Tv, self.llm_hidden_size)
            # Global average pool video temporal dimension to fuse with graph tokens
            v_context = torch.mean(v_embeds, dim=1, keepdim=True) # (B, 1, llm_hidden_size)
            multimodal_tokens = multimodal_tokens + v_context      # Broadcasting cross-modal fusion

        # Step 4: Mean pooling across nodes for macro-level context representation
        macro_tokens = torch.mean(multimodal_tokens, dim=1)  # (B, llm_hidden_size)

        # Step 5: Multi-Task Predictions
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
    print("Testing MultimodalTransportationMLLM with Multimodal Visual Inputs...")
    
    model = MultimodalTransportationMLLM(
        in_channels=2,
        graph_embed_dim=128,
        num_nodes=10,
        pred_steps=12,
        use_vision_tokens=True
    )

    mock_x = torch.randn(4, 12, 10, 2)  # (Batch=4, T_in=12, Nodes=10, Feats=2)
    mock_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    mock_edge_weight = torch.tensor([1.0, 0.8, 0.5, 0.9], dtype=torch.float32)
    mock_vision = torch.randn(4, 5, 3, 224, 224) # (Batch=4, Frames=5, C=3, H=224, W=224)

    outputs = model(mock_x, mock_edge_index, mock_edge_weight, vision_tensor=mock_vision)
    print(f"Meso Traffic Flow shape:   {outputs['meso_flow'].shape}")        # [4, 12, 10, 2]
    print(f"Macro Travel Demand shape: {outputs['macro_demand'].shape}")     # [4, 12]
    print(f"Multimodal Token Space:    {outputs['multimodal_tokens'].shape}") # [4, 10, 3584]