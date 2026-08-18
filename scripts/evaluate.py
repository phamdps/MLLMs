"""
Quantitative Evaluation Script for Multimodal Transportation Digital Twin MLLM
Computes MAE, RMSE, and MAPE on the METR-LA test dataset split.
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

import os
import sys
from pathlib import Path
# Add project root to Python path dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.dataloader.load_real_data import prepare_metr_la_datasets
from src.dataloader.dataset import multimodal_collate_fn
from src.models.backbone import MultimodalTransportationMLLM



def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    """
    Computes MAE, RMSE, and MAPE between predictions and ground truth.
    Handles small values for MAPE to avoid division by zero.
    """
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    # Avoid division by zero in MAPE
    mask = targets > 1e-5
    if np.any(mask):
        mape = np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100
    else:
        mape = 0.0
        
    return mae, rmse, mape


def evaluate(config_path: str, checkpoint_path: str):
    # 1. Load Configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Running METR-LA Quantitative Evaluation on device: {device}")

    data_cfg = config.get("data", {})
    h5_file = data_cfg.get("h5_path", "data/raw/METR-LA/metr_la.h5")
    pkl_file = data_cfg.get("pkl_path", "data/raw/METR-LA/adj_mx.pkl")
    history_steps = data_cfg.get("history_steps", 12)
    pred_steps = data_cfg.get("pred_steps", 12)
    
    batch_size = config.get("training", {}).get("batch_size", 16)

    # 2. Prepare Datasets & Test DataLoader
    print("Loading METR-LA test dataset split...")
    datasets = prepare_metr_la_datasets(
        h5_path=h5_file,
        pkl_path=pkl_file,
        history_steps=history_steps,
        pred_steps=pred_steps
    )

    test_dataset = datasets.get("test", datasets["val"])  # Fallback to val if test not explicitly defined
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=2
    )

    # 3. Load Trained Model Checkpoint
    num_nodes = test_dataset.raw_data.shape[1]
    in_channels = test_dataset.raw_data.shape[2]

    model = MultimodalTransportationMLLM(
        in_channels=in_channels,
        graph_embed_dim=128,
        num_nodes=num_nodes,
        pred_steps=pred_steps,
        llm_hidden_size=3584,
        use_vision_tokens=True
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")

    print(f"Loading weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Evaluation Loop
    all_preds = []
    all_targets = []

    print("Evaluating model across test batches...")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            graph = batch["graph"].to(device)
            
            vision = batch.get("vision", None)
            if vision is not None:
                vision = vision.to(device)

            # Forward pass
            outputs = model(
                x=x,
                edge_index=graph.edge_index,
                edge_weight=graph.edge_attr,
                vision_tensor=vision
            )

            pred_flow = outputs["meso_flow"]  # (B, T_out, N, F)

            all_preds.append(pred_flow.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate all batches
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # 5. Compute and Print Metrics
    mae, rmse, mape = compute_metrics(preds_arr, targets_arr)

    print("\n" + "="*50)
    print("📈 METR-LA Quantitative Evaluation Results:")
    print("="*50)
    print(f"  • Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"  • Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"  • Mean Absolute Percentage (MAPE): {mape:.2f}%")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_transportation_mllm.pt")
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint)

# python3 scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_transportation_mllm.pt
