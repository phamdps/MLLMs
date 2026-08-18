import torch
import torch.nn as nn
from src.models.encoders import SpatialGraphEncoder

class MultimodalDigitalTwinInferenceModel(nn.Module):
    def __init__(self, num_sensors=207, num_zones=2, num_agents=2, in_steps=12, hidden_dim=64, out_steps=12):
        super().__init__()
        self.out_steps = out_steps
        self.num_sensors = num_sensors
        self.num_zones = num_zones
        self.num_agents = num_agents
        
        # Spatial-Temporal Graph Encoder
        self.spatial_encoder = SpatialGraphEncoder(
            in_channels=1, hidden_dim=hidden_dim, out_dim=hidden_dim
        )
        
        # Flattened spatial representation dimension after graph pooling
        encoder_out_dim = num_sensors * hidden_dim
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Task Heads
        self.meso_head = nn.Linear(hidden_dim, out_steps * num_sensors * 1)
        self.macro_head = nn.Linear(hidden_dim, out_steps)
        self.traj_head = nn.Linear(hidden_dim, num_agents * out_steps * 2)
        self.od_head = nn.Linear(hidden_dim, out_steps * num_zones * num_zones)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: Input tensor of shape (Batch, Time_in, Sensors, Features)
        edge_index: Graph topology edge indices (2, E)
        """
        batch_size = x.size(0)
        
        # Pass through ST-GNN Encoder -> Shape: (B, N, hidden_dim)
        h_graph = self.spatial_encoder(x, edge_index, edge_weight)
        
        # Flatten node representations for global task projection -> Shape: (B, N * hidden_dim)
        h_flat = h_graph.reshape(batch_size, -1)
        h = self.shared_mlp(h_flat) # Shape: (B, hidden_dim)
        
        # Predictors
        pred_meso = self.meso_head(h).view(batch_size, self.out_steps, self.num_sensors, 1)
        pred_macro = self.macro_head(h).view(batch_size, self.out_steps)
        pred_traj = self.traj_head(h).view(batch_size, self.num_agents, self.out_steps, 2)
        pred_od = self.od_head(h).view(batch_size, self.out_steps, self.num_zones, self.num_zones)
        
        return {
            "meso": pred_meso,
            "macro": pred_macro,
            "traj": pred_traj,
            "od": pred_od
        }

class MultimodalDigitalTwinModel(nn.Module):
    def __init__(self, num_sensors=207, num_zones=2, num_agents=2, in_steps=12, hidden_dim=64, out_steps=12):
        super().__init__()
        self.out_steps = out_steps
        self.num_sensors = num_sensors
        self.num_zones = num_zones
        self.num_agents = num_agents
        
        # Calculate correct flattened input dimension: Time_in * Sensors * Features
        in_features = in_steps * num_sensors * 1
        
        # Shared Spatiotemporal Backbone
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Task Heads
        self.meso_head = nn.Linear(hidden_dim, out_steps * num_sensors * 1)
        self.macro_head = nn.Linear(hidden_dim, out_steps)
        self.traj_head = nn.Linear(hidden_dim, num_agents * out_steps * 2)
        self.od_head = nn.Linear(hidden_dim, out_steps * num_zones * num_zones)

    def forward(self, x):
        """
        x: Input tensor of shape (Batch, Time_in, Sensors, Features)
        """
        batch_size = x.size(0)
        
        # Flatten spatial/temporal input: (B, T_in, N, F) -> (B, T_in * N * F)
        x_flat = x.reshape(batch_size, -1)
        h = self.shared_encoder(x_flat) # Shape: (B, hidden_dim)
        
        # Predictors
        pred_meso = self.meso_head(h).view(batch_size, self.out_steps, self.num_sensors, 1)
        pred_macro = self.macro_head(h).view(batch_size, self.out_steps)
        pred_traj = self.traj_head(h).view(batch_size, self.num_agents, self.out_steps, 2)
        pred_od = self.od_head(h).view(batch_size, self.out_steps, self.num_zones, self.num_zones)
        
        return {
            "meso": pred_meso,
            "macro": pred_macro,
            "traj": pred_traj,
            "od": pred_od
        }

def compute_multitask_loss(preds, targets, weights=None):
    """
    Computes the joint weighted multi-task loss matching our mathematical formulation.
    """
    if weights is None:
        weights = {"meso": 1.0, "macro": 0.5, "traj": 1.0, "od": 0.5}
        
    mse_loss = nn.MSELoss()
    
    # Task A Loss: Meso-Flow
    l_meso = mse_loss(preds["meso"], targets["meso"])
    
    # Task B Loss: Macro-Demand
    l_macro = mse_loss(preds["macro"], targets["macro"])
    
    # Task C Loss: Trajectory
    l_traj = mse_loss(preds["traj"], targets["traj"])
    
    # Task D Loss: OD Demand
    l_od = mse_loss(preds["od"], targets["od"])
    
    # Total Joint Loss
    l_total = (
        weights["meso"] * l_meso +
        weights["macro"] * l_macro +
        weights["traj"] * l_traj +
        weights["od"] * l_od
    )
    
    return l_total, {
        "l_meso": l_meso.item(),
        "l_macro": l_macro.item(),
        "l_traj": l_traj.item(),
        "l_od": l_od.item()
    }