import torch
import torch.nn as nn

class MultimodalUnifiedEmbedding(nn.Module):
    def __init__(self, num_sensors=207, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Numerical Time-Series Encoder (e.g., simple linear/conv projection)
        self.ts_encoder = nn.Linear(1, hidden_dim)
        
        # 2. Graph Spatial Mock Encoder (Simulating GCN node embedding)
        self.graph_encoder = nn.Linear(num_sensors, hidden_dim)
        
        # 3. Vision Projection (Assuming input image features come from a CNN/ViT backbone of dim 512)
        self.vision_proj = nn.Linear(512, hidden_dim)
        
        # 4. Text Projection (Assuming input text embeddings come from a LLM of dim 768)
        self.text_proj = nn.Linear(768, hidden_dim)
        
        # 5. Cross-Modal Attention for Fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, ts_data, graph_data, image_feats, text_feats):
        """
        ts_data:    (Batch, Time_in, Sensors, Features=1)
        graph_data: (Batch, Sensors, Num_Nodes)
        image_feats:(Batch, 512) -> e.g., pooled CNN/ViT output
        text_feats: (Batch, 768) -> e.g., pooled text encoder output
        """
        B = ts_data.size(0)
        
        # Step A: Project all modalities into the unified hidden_dim space (d = hidden_dim)
        h_ts = self.ts_encoder(ts_data).mean(dim=1)     # Shape: (B, Sensors, hidden_dim)
        h_graph = self.graph_encoder(graph_data)        # Shape: (B, Sensors, hidden_dim)
        
        h_vision = self.vision_proj(image_feats).unsqueeze(1) # Shape: (B, 1, hidden_dim)
        h_text = self.text_proj(text_feats).unsqueeze(1)       # Shape: (B, 1, hidden_dim)
        
        # Step B: Combine baseline numerical & graph features as our core Spatiotemporal Context
        core_spatiotemporal = h_ts + h_graph            # Shape: (B, Sensors, hidden_dim)
        
        # Step C: Treat external modalities (Vision + Text) as auxiliary Context tokens
        external_context = torch.cat([h_vision, h_text], dim=1) # Shape: (B, 2, hidden_dim)
        
        # Step D: Cross-Modal Alignment via Attention 
        # Query = Core Spatiotemporal traffic state, Key/Value = External Vision & Text context
        fused_features, _ = self.cross_attention(
            query=core_spatiotemporal, 
            key=external_context, 
            value=external_context
        )
        
        # Final unified representation combining traffic state + cross-modal guidance
        unified_representation = core_spatiotemporal + fused_features # Shape: (B, Sensors, hidden_dim)
        
        return unified_representation