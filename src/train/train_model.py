"""
End-to-End Training Script for Multimodal Transportation Digital Twin MLLM
Config-driven via YAML with support for spatial graph embeddings and multimodal inputs.
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.dataloader.load_real_data import prepare_metr_la_datasets
from src.dataloader.dataset import multimodal_collate_fn
from src.models.backbone import MultimodalTransportationMLLM
from src.evaluation.multi_task_loss import MultiTaskTransportationLoss


def train(config_path: str):
    # 1. Load YAML Configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Multimodal Spatiotemporal Training on device: {device}")

    # Extract configurations safely
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", config.get("config", {}))
    
    h5_file = data_cfg.get("h5_path", "data/raw/METR-LA/metr_la.h5")
    pkl_file = data_cfg.get("pkl_path", "data/raw/METR-LA/adj_mx.pkl")
    history_steps = data_cfg.get("history_steps", 12)
    pred_steps = data_cfg.get("pred_steps", 12)
    
    batch_size = train_cfg.get("batch_size", 16)
    epochs = train_cfg.get("epochs", 10)
    lr = float(train_cfg.get("learning_rate", 1e-4))
    save_dir = train_cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # 2. Prepare Datasets & DataLoaders
    print("Loading METR-LA datasets and spatial graph topology...")
    datasets = prepare_metr_la_datasets(
        h5_path=h5_file,
        pkl_path=pkl_file,
        history_steps=history_steps,
        pred_steps=pred_steps
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=2
    )

    # 3. Instantiate Model Backbone & Uncertainty Loss
    num_nodes = datasets["train"].raw_data.shape[1]   # Typically 207 for METR-LA
    in_channels = datasets["train"].raw_data.shape[2] # Typically feature channels (1 or 2)

    model = MultimodalTransportationMLLM(
        in_channels=in_channels,
        graph_embed_dim=128,
        num_nodes=num_nodes,
        pred_steps=pred_steps,
        llm_hidden_size=3584,
        use_vision_tokens=True
    ).to(device)

    criterion = MultiTaskTransportationLoss().to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Setup completed successfully. Starting training loops...")
    best_val_loss = float("inf")

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device)           # (B, T_in, N, F)
            y = batch["y"].to(device)           # (B, T_out, N, F)
            graph = batch["graph"].to(device)   # PyG Batch object
            
            vision = batch.get("vision", None)
            if vision is not None:
                vision = vision.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                x=x,
                edge_index=graph.edge_index,
                edge_weight=graph.edge_attr,
                vision_tensor=vision
            )

            pred_flow = outputs["meso_flow"]
            true_demand = torch.mean(y, dim=(2, 3))
            pred_demand = outputs["macro_demand"]

            # Compute joint uncertainty loss
            loss, metrics = criterion(pred_flow, y, pred_demand, true_demand)

            # Backpropagation & Step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | "
                    f"Total Loss: {loss.item():.4f} | Flow Loss: {metrics['flow_loss']:.4f} | "
                    f"Demand Loss: {metrics['demand_loss']:.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()

        # 5. Validation Loop
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                graph = batch["graph"].to(device)
                vision = batch.get("vision", None)
                if vision is not None:
                    vision = vision.to(device)

                outputs = model(
                    x=x,
                    edge_index=graph.edge_index,
                    edge_weight=graph.edge_attr,
                    vision_tensor=vision
                )

                pred_flow = outputs["meso_flow"]
                true_demand = torch.mean(y, dim=(2, 3))
                pred_demand = outputs["macro_demand"]

                val_loss, _ = criterion(pred_flow, y, pred_demand, true_demand)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"✨ --- Epoch {epoch} Finished --- Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(save_dir, "best_transportation_mllm.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, ckpt_path)
            print(f"💾 Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    train(args.config)