# This module generates and applies physical spatial mapping matrices ($A_{\text{meso} \rightarrow \text{macro}} \in \mathbb{R}^{M \times N}$) to aggregate $N$ fine-grained meso sensor nodes (e.g., METR-LA sensors) into $M$ macro zones (e.g., city districts or grid cells).

import torch
import numpy as np
from typing import Union, List, Optional


class SpatialAggregationMatrix:
    """
    Constructs and manages spatial mapping matrices A_{meso -> macro} to project
    fine-grained meso-level node sensor streams to macro-level zonal volumes.
    
    Physical Consistency Rule:
        Macro_Demand = A_{meso -> macro} @ Meso_Flow
    """
    def __init__(
        self,
        num_meso_nodes: int,
        num_macro_zones: int,
        assignment_matrix: Optional[np.ndarray] = None,
        normalize: str = "row"
    ):
        """
        Args:
            num_meso_nodes (int): N fine-grained sensors/nodes.
            num_macro_zones (int): M coarse zones/districts.
            assignment_matrix (np.ndarray, optional): Binary/weighted mapping matrix of shape (M, N).
            normalize (str): Matrix normalization scheme: 'row' (average), 'sum' (unweighted), or 'none'.
        """
        self.num_meso_nodes = num_meso_nodes
        self.num_macro_zones = num_macro_zones
        self.normalize = normalize

        if assignment_matrix is not None:
            assert assignment_matrix.shape == (num_macro_zones, num_meso_nodes), \
                f"Expected shape ({num_macro_zones}, {num_meso_nodes}), got {assignment_matrix.shape}"
            self.matrix = torch.tensor(assignment_matrix, dtype=torch.float32)
        else:
            # Fallback: Construct synthetic uniform spatial mapping matrix
            self.matrix = self._generate_synthetic_mapping()

        if normalize == "row":
            row_sums = self.matrix.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0  # Prevent division by zero
            self.matrix = self.matrix / row_sums

    def _generate_synthetic_mapping(self) -> torch.Tensor:
        """
        Creates a synthetic, spatially contiguous assignment matrix mapping
        N sensors evenly into M macro zones.
        """
        mapping = torch.zeros((self.num_macro_zones, self.num_meso_nodes), dtype=torch.float32)
        nodes_per_zone = self.num_meso_nodes // self.num_macro_zones
        
        for zone_idx in range(self.num_macro_zones):
            start_node = zone_idx * nodes_per_zone
            end_node = (zone_idx + 1) * nodes_per_zone if zone_idx < self.num_macro_zones - 1 else self.num_meso_nodes
            mapping[zone_idx, start_node:end_node] = 1.0

        return mapping

    def aggregate_flow(self, meso_flow: torch.Tensor) -> torch.Tensor:
        """
        Applies spatial mapping matrix to aggregate meso-level flows into macro zonal demand.

        Args:
            meso_flow (torch.Tensor): Tensor of shape (Batch, Steps, Num_Meso_Nodes, Features) 
                                      or (Batch, Num_Meso_Nodes)

        Returns:
            torch.Tensor: Aggregated macro demand of shape (Batch, Steps, Num_Macro_Zones) 
                          or (Batch, Num_Macro_Zones)
        """
        matrix_device = self.matrix.to(meso_flow.device)

        if meso_flow.dim() == 4:
            # Shape: (B, T, N, F) -> select main feature channel (e.g., speed/volume at F=0)
            flow_channel = meso_flow[..., 0]  # (B, T, N)
            # Matrix multiply along spatial dimension: (B, T, M) = (B, T, N) @ (N, M)
            macro_demand = torch.matmul(flow_channel, matrix_device.T)
        elif meso_flow.dim() == 2:
            # Shape: (B, N) -> (B, M)
            macro_demand = torch.matmul(meso_flow, matrix_device.T)
        else:
            raise ValueError(f"Unsupported tensor dimension for meso_flow: {meso_flow.dim()}")

        return macro_demand

    def get_matrix_tensor(self) -> torch.Tensor:
        """Returns the spatial mapping tensor A_{meso -> macro}."""
        return self.matrix


# --- Sanity Check Script ---
if __name__ == "__main__":
    print("Testing SpatialAggregationMatrix...")
    
    N_nodes = 10   # Meso sensors
    M_zones = 3    # Macro zones
    
    aggregator = SpatialAggregationMatrix(num_meso_nodes=N_nodes, num_macro_zones=M_zones, normalize="row")
    
    # Mock meso flow tensor: (Batch=4, T_out=12, Nodes=10, Features=2)
    mock_meso_flow = torch.randn(4, 12, N_nodes, 2)
    
    macro_demand = aggregator.aggregate_flow(mock_meso_flow)
    print(f"Mapping Matrix Shape A_(meso->macro): {aggregator.matrix.shape}")  # [3, 10]
    print(f"Aggregated Macro Demand Shape:      {macro_demand.shape}")         # [4, 12, 3]


