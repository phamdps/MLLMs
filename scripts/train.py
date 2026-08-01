import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.dataloader.load_real_data import prepare_metr_la_datasets
from src.models.backbone import MultimodalTransportationMLLM
from src.models.multi_task_loss import MultiTaskUncertaintyLoss


def train(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running MLLM Spatiotemporal Training on device: {device}")

    # Load real datasets
    h5_file = "data/raw/METR-LA/metr_la.h5"
    pkl_file = "data/raw/METR-LA/adj_mx.pkl"
    
    datasets = prepare_metr_la_datasets(
        h5_path=h5_file,
        pkl_path=pkl_file,
        history_steps=config["data"]["history_steps"],
        pred_steps=config["data"]["pred_steps"]
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # Instantiate model & uncertainty loss balancer
    model = MultimodalTransportationMLLM(
        in_channels=1,
        graph_embed_dim=128,
        num_nodes=207,
        pred_steps=config["data"]["pred_steps"]
    ).to(device)

    loss_balancer = MultiTaskUncertaintyLoss(num_tasks=2).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=float(config["training"]["learning_rate"])
    )

    print("Setup completed successfully. Ready for training loops.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    train(args.config)