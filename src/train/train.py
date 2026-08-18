"""
End-to-End Training Script for Multimodal Transportation Digital Twin MLLM
"""

import os
import torch
from torch.utils.data import DataLoader
from src.dataloader.load_real_data import prepare_metr_la_datasets
from src.dataloader.dataset import multimodal_collate_fn
from src.models.backbone import MultimodalTransportationMLLM
from src.evaluation.multi_task_loss import MultiTaskTransportationLoss

def train_transportation_mllm(
    h5_path: str = "data/raw/METR-LA/metr_la.h5",
    pkl_path: str = "data/raw/METR-LA/adj_mx.pkl",
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints"
):
    """
    Production-grade training loop for the Multimodal Transportation MLLM.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"🚀 Initializing training pipeline on device: {device}")

    # 1. Prepare Datasets & DataLoaders
    print("Loading METR-LA dataset and graph topology...")
    data_dicts = prepare_metr_la_datasets(
        h5_path=h5_path,
        pkl_path=pkl_path,
        history_steps=12,
        pred_steps=12
    )

    train_dataset = data_dicts["train"]
    val_dataset = data_dicts["val"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=2
    )

    # 2. Instantiate Model Backbone
    print("Building Multimodal Transportation MLLM backbone...")
    model = MultimodalTransportationMLLM(
        in_channels=2,
        graph_embed_dim=128,
        num_nodes=train_dataset.raw_data.shape[1],  # Standard sensor nodes (e.g., 207)
        pred_steps=12,
        llm_hidden_size=3584,
        use_vision_tokens=True
    ).to(device)

    # 3. Setup Multi-Task Uncertainty Loss & Optimizer
    criterion = MultiTaskTransportationLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=lr, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device)           # (B, T_in, N, F)
            y = batch["y"].to(device)           # (B, T_out, N, F)
            graph = batch["graph"].to(device)   # PyG Batch object
            
            # Optional vision frames if present in batch
            vision = batch.get("vision", None)
            if vision is not None:
                vision = vision.to(device)

            optimizer.zero_grad()

            # Forward pass through Multimodal MLLM Backbone
            outputs = model(
                x=x,
                edge_index=graph.edge_index,
                edge_weight=graph.edge_attr,
                vision_tensor=vision
            )

            pred_flow = outputs["meso_flow"]         # (B, T_out, N, F)
            # Generate synthetic macro demand target from ground truth flow for training demonstration
            true_demand = torch.mean(y, dim=(2, 3))  # (B, T_out)
            pred_demand = outputs["macro_demand"]    # (B, T_out)

            # Compute multi-task joint uncertainty loss
            loss, metrics = criterion(pred_flow, y, pred_demand, true_demand)

            # Backpropagation & Optimization step
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
        print(f"✨ --- Epoch {epoch} Completed --- Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

        # Save Best Model Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(save_dir, "best_transportation_mllm.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, ckpt_path)
            print(f"💾 Saved new best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train_transportation_mllm(
        h5_path="data/raw/METR-LA/metr_la.h5",
        pkl_path="data/raw/METR-LA/adj_mx.pkl",
        batch_size=16,
        epochs=3,
        lr=2e-4
    )