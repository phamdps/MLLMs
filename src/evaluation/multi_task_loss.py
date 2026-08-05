import torch
import torch.nn as nn
from typing import Dict, Any


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
        """
        Args:
            meso_weight (float): Weight for microscopic sensor prediction loss.
            macro_weight (float): Weight for macroscopic zonal demand prediction loss.
            loss_type (str): Regression loss type ('mse' or 'smooth_l1').
        """
        super().__init__()
        self.meso_weight = meso_weight
        self.macro_weight = macro_weight

        if loss_type.lower() == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the combined weighted multi-task loss.

        Args:
            predictions (Dict): Model output containing 'meso_flow' and 'macro_demand'.
            targets (Dict or torch.Tensor): Ground truth targets. If a tensor is provided, 
                                            it treats it as meso targets and derives macro targets.

        Returns:
            Dict containing individual loss metrics and the combined total loss.
        """
        # 1. Extract predictions
        pred_meso = predictions["meso_flow"]      # (B, T_out, N, F)
        pred_macro = predictions["macro_demand"]  # (B, T_out)

        # 2. Extract or derive ground truth targets
        if isinstance(targets, dict):
            true_meso = targets["y"]              # (B, T_out, N, F)
            # Derive macro demand ground truth by summing true sensor values across nodes (Feature channel 0)
            true_macro = true_meso[..., 0].sum(dim=-1) # (B, T_out)
        else:
            true_meso = targets
            true_macro = true_meso[..., 0].sum(dim=-1)

        # 3. Compute individual task losses
        meso_loss = self.criterion(pred_meso, true_meso)
        macro_loss = self.criterion(pred_macro, true_macro)

        # 4. Compute weighted total loss
        total_loss = (self.meso_weight * meso_loss) + (self.macro_weight * macro_loss)

        return {
            "total_loss": total_loss,
            "meso_loss": meso_loss.detach(),
            "macro_loss": macro_loss.detach()
        }


# --- Sanity Check Script ---
if __name__ == "__main__":
    print("Testing MultiTaskTransportationLoss...")
    
    loss_fn = MultiTaskTransportationLoss(meso_weight=1.0, macro_weight=0.5)
    
    # Mock predictions
    mock_preds = {
        "meso_flow": torch.randn(4, 12, 10, 2, requires_grad=True),
        "macro_demand": torch.randn(4, 12, requires_grad=True)
    }
    
    # Mock targets
    mock_targets = {
        "y": torch.randn(4, 12, 10, 2)
    }
    
    losses = loss_fn(mock_preds, mock_targets)
    print(f"Total Combined Loss: {losses['total_loss'].item():.4f}")
    print(f"Meso-flow Loss:      {losses['meso_loss'].item():.4f}")
    print(f"Macro-demand Loss:   {losses['macro_loss'].item():.4f}")