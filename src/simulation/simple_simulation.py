import torch

def run_transportation_simulation():
    print("🚦 Starting Transportation Digital Twin Simulation...\n")

    # Parameters
    batch_size = 2       # 2 different scenarios (e.g., Batch 0: Normal Day, Batch 1: Heavy Commute)
    num_steps = 12       # 12 future time steps (5-minute intervals = 1 hour)
    num_nodes = 207      # METR-LA sensor network size
    features = 1         # Traffic flow / speed feature

    # 1. Simulate Ground Truth (True Traffic Data)
    # Let's say true sensor flows are random values between 10 and 100 vehicles per interval
    true_meso_flow = torch.randint(10, 100, (batch_size, num_steps, num_nodes, features), dtype=torch.float32)

    # Derive Macro Demand ground truth by summing sensor flows across all 207 nodes (Feature channel 0)
    true_macro_demand = true_meso_flow[..., 0].sum(dim=-1) # Shape: (Batch, Time)

    print(f"📊 [Ground Truth Generated]")
    print(f"   - Meso Flow Shape (Sensor Level): {true_meso_flow.shape}")
    print(f"   - Macro Demand Shape (Zonal Total): {true_macro_demand.shape}")
    print(f"   - Sample Zonal Total at Step 0 (Batch 0): {true_macro_demand[0, 0].item():.1f} vehicles\n")

    # 2. Simulate Model Predictions (With slight simulated prediction noise)
    pred_meso_flow = true_meso_flow + torch.randn_like(true_meso_flow) * 5.0
    pred_macro_demand = true_macro_demand + torch.randn_like(true_macro_demand) * 50.0

    print(f"🤖 [Model Predictions Generated]")
    print(f"   - Predicted Meso Shape: {pred_meso_flow.shape}")
    print(f"   - Predicted Macro Shape: {pred_macro_demand.shape}")
    print(f"   - Sample Predicted Zonal Total at Step 0 (Batch 0): {pred_macro_demand[0, 0].item():.1f} vehicles\n")

    # 3. Compute Multi-Task Losses
    criterion = torch.nn.MSELoss()

    meso_loss = criterion(pred_meso_flow, true_meso_flow)
    macro_loss = criterion(pred_macro_demand, true_macro_demand)

    # Weighted combination (e.g., meso_weight=1.0, macro_weight=0.5)
    meso_weight = 1.0
    macro_weight = 0.5
    total_loss = (meso_weight * meso_loss) + (macro_weight * macro_loss)

    print(f"📉 [Loss Evaluation Results]")
    print(f"   - Task A (Meso-flow Loss):   {meso_loss.item():.4f}")
    print(f"   - Task B (Macro-demand Loss): {macro_loss.item():.4f}")
    print(f"   - Combined Multi-Task Loss:   {total_loss.item():.4f}\n")

    # 4. Scenario Insight
    print("💡 What this means for our Digital Twin:")
    print("   -> The Meso loss penalizes errors at individual sensors (e.g., missing a bottleneck on I-5).")
    print("   -> The Macro loss penalizes errors in city-wide traffic volume (e.g., underestimating total inbound commuters).")
    print("   -> Joint optimization forces the AI to respect both local bottlenecks and regional capacity constraints!")

if __name__ == "__main__":
    run_transportation_simulation()