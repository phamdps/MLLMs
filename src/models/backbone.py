import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.encoders import SpatialGraphEncoder


class CrossModalProjector(nn.Module):
    """
    Connects graph encoder embeddings to the LLM token embedding space.
    """
    def __init__(self, encoder_dim: int = 128, llm_dim: int = 2048):
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
    Unifies Spatial Graph Neural Encoders, Cross-Modal Connectors, and LLM Decoders.
    """
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        in_channels: int = 2,
        graph_embed_dim: int = 128,
        num_nodes: int = 10,
        pred_steps: int = 12,
        use_quantization: bool = False
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_steps = pred_steps

        # 1. Spatial Graph Encoder
        self.graph_encoder = SpatialGraphEncoder(
            in_channels=in_channels,
            hidden_dim=64,
            out_dim=graph_embed_dim
        )

        # 2. Cross-Modal Projector (Connector)
        # Assuming typical LLM hidden size (e.g., 2048 or 4096)
        self.llm_hidden_size = 2048
        self.projector = CrossModalProjector(
            encoder_dim=graph_embed_dim,
            llm_dim=self.llm_hidden_size
        )

        # 3. Multi-Task Prediction Decoders (Meso-flow & Macro-demand)
        self.meso_flow_head = nn.Sequential(
            nn.Linear(self.llm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, pred_steps * in_channels)
        )

        self.macro_demand_head = nn.Sequential(
            nn.Linear(self.llm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, pred_steps)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal joint forecasting.

        Args:
            x: Sensor time-series tensor (B, T_in, N, F)
            edge_index: Graph adjacency topology (2, E)
            input_ids: Tokenized text prompt IDs (B, Seq_Len)
            attention_mask: Attention mask for text prompts (B, Seq_Len)

        Returns:
            Dict containing predicted 'meso_flow' and 'macro_demand' tensors.
        """
        B, T_in, N, F_dim = x.shape

        # Step 1: Extract spatial graph embeddings
        graph_feats = self.graph_encoder(x, edge_index, edge_weight)  # (B, N, graph_embed_dim)

        # Step 2: Project spatial embeddings into LLM cross-modal space
        multimodal_tokens = self.projector(graph_feats)  # (B, N, llm_hidden_size)

        # Step 3: Mean pooling across nodes for macro-level context representation
        macro_tokens = torch.mean(multimodal_tokens, dim=1)  # (B, llm_hidden_size)

        # Step 4: Multi-Task Predictions
        # Meso-level flow prediction: per-node future steps (B, T_out, N, F)
        meso_out = self.meso_flow_head(multimodal_tokens)  # (B, N, T_out * F)
        meso_out = meso_out.view(B, N, self.pred_steps, F_dim).permute(0, 2, 1, 3)

        # Macro-level travel demand prediction: zonal volume (B, T_out)
        macro_out = self.macro_demand_head(macro_tokens)  # (B, T_out)

        return {
            "meso_flow": meso_out,
            "macro_demand": macro_out
        }


# --- Sanity Check Script ---
if __name__ == "__main__":
    print("Testing MultimodalTransportationMLLM Backbone...")
    
    model = MultimodalTransportationMLLM(
        in_channels=2,
        graph_embed_dim=128,
        num_nodes=10,
        pred_steps=12
    )

    mock_x = torch.randn(8, 12, 10, 2)  # (Batch=8, T_in=12, Nodes=10, Feats=2)
    mock_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    outputs = model(mock_x, mock_edge_index)
    print(f"Meso Traffic Flow shape:   {outputs['meso_flow'].shape}")     # [8, 12, 10, 2]
    print(f"Macro Travel Demand shape: {outputs['macro_demand'].shape}")  # [8, 12]