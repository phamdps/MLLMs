import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, Tuple


class MultiTaskTransportationLoss(nn.Module):
    """
    Multi-Task Loss Function for Transportation Digital Twin.
    Balances fine-grained sensor time-series forecasting (Meso-flow) 
    with coarse-grained regional demand forecasting (Macro-demand).
    """
    def __init__(
        self,
        meso_weight: float = 1.0,
        macro_weight: float = 0.5,
        loss_type: str = "mse"
    ):
        super().__init__()
        self.meso_weight = meso_weight
        self.macro_weight = macro_weight

        if loss_type.lower() == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

    def forward(
        self,
        preds_or_flow: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets_or_y: Union[Dict[str, torch.Tensor], torch.Tensor],
        pred_demand: Optional[torch.Tensor] = None,
        true_demand: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the combined weighted multi-task loss. Supports both dictionary inputs 
        and legacy 4-argument positional calls from training loops.
        """
        if not isinstance(preds_or_flow, dict):
            pred_meso = preds_or_flow
            true_meso = targets_or_y
            
            if pred_demand is None:
                pred_macro = pred_meso[..., 0].sum(dim=-1)
            else:
                pred_macro = pred_demand
                
            if true_demand is None:
                true_macro = true_meso[..., 0].sum(dim=-1)
            else:
                true_macro = true_demand
        else:
            pred_meso = preds_or_flow["meso_flow"]
            pred_macro = preds_or_flow["macro_demand"]

            if isinstance(targets_or_y, dict):
                true_meso = targets_or_y["y"]
                true_macro = true_meso[..., 0].sum(dim=-1)
            else:
                true_meso = targets_or_y
                true_macro = true_meso[..., 0].sum(dim=-1)

        # 3. Compute individual task losses
        meso_loss = self.criterion(pred_meso, true_meso)
        macro_loss = self.criterion(pred_macro, true_macro)

        # 4. Compute weighted total loss
        total_loss = (self.meso_weight * meso_loss) + (self.macro_weight * macro_loss)

        # Metrics matching training script expectations (using 'flow_loss' and 'demand_loss')
        metrics = {
            "flow_loss": meso_loss.item(),
            "demand_loss": macro_loss.item(),
            "meso_loss": meso_loss.item(),
            "macro_loss": macro_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, metrics