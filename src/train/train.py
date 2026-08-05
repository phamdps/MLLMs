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
    End-to-end training loop for the Multimodal Transportation Digital Twin MLLM.
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
        num_nodes=train_dataset.raw_data.shape[1],  # Number of sensors (e.g., 207 for METR-LA)
        pred_steps=12,
        llm_hidden_size=3584
    ).to(device)

    # 3. Setup Loss Function and Optimizer
    criterion = MultiTaskTransportationLoss(meso_weight=1.0, macro_weight=0.5, loss_type="mse")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to device
            x = batch["x"].to(device)           # (B, T_in, N, F)
            y = batch["y"].to(device)           # (B, T_out, N, F)
            graph = batch["graph"].to(device)   # PyG Batch object

            optimizer.zero_grad()

            # Forward pass through backbone
            predictions = model(
                x=x,
                edge_index=graph.edge_index,
                edge_weight=graph.edge_attr
            )

            # Compute multi-task losses
            loss_dict = criterion(predictions, {"y": y})
            loss = loss_dict["total_loss"]

            # Backpropagation & Optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | "
                    f"Train Loss: {loss.item():.4f} (Meso: {loss_dict['meso_loss'].item():.4f}, "
                    f"Macro: {loss_dict['macro_loss'].item():.4f})"
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

                predictions = model(
                    x=x,
                    edge_index=graph.edge_index,
                    edge_weight=graph.edge_attr
                )

                loss_dict = criterion(predictions, {"y": y})
                total_val_loss += loss_dict["total_loss"].item()

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
    # Example execution handle
    train_transportation_mllm(
        h5_path="data/raw/METR-LA/metr_la.h5",
        pkl_path="data/raw/METR-LA/adj_mx.pkl",
        batch_size=16,
        epochs=5,
        lr=2e-4
    )