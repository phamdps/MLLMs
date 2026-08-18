import torch
from src.models.multitask_dt import MultimodalDigitalTwinModel, compute_multitask_loss

def run_tensor_verification():
    print("🚀 Initializing Multimodal Digital Twin Model...")
    
    # Define dimensions matching our METR-LA / digital twin setup
    B, T_in, N, F = 2, 12, 207, 1
    T_out = 12
    Z = 2 # 2 Zones for OD
    A = 2 # 2 Agents for Trajectory
    
    model = MultimodalDigitalTwinModel(
        num_sensors=N, num_zones=Z, num_agents=A, hidden_dim=64, out_steps=T_out
    )
    model.eval() # or train()
    
    # Create synthetic live input tensor (Batch, Time_in, Sensors, Features)
    dummy_input = torch.randn(B, T_in, N, F)
    print(f"📥 Input Tensor Shape: {dummy_input.shape}")
    
    # Forward pass
    preds = model(dummy_input)
    print("\n✅ Forward Pass Successful! Prediction Shapes:")
    for task_name, tensor in preds.items():
        print(f"   - {task_name.upper()}: {tensor.shape}")
        
    # Construct corresponding ground truth targets matching shapes
    targets = {
        "meso": torch.randn(B, T_out, N, 1),
        "macro": torch.randn(B, T_out),
        "traj": torch.randn(B, A, T_out, 2),
        "od": torch.randn(B, T_out, Z, Z)
    }
    
    # Compute joint loss
    total_loss, loss_dict = compute_multitask_loss(preds, targets)
    
    print("\n📊 Multi-Task Joint Loss Breakdown:")
    print(f"   - Meso Loss (Task A):   {loss_dict['l_meso']:.4f}")
    print(f"   - Macro Loss (Task B):  {loss_dict['l_macro']:.4f}")
    print(f"   - Trajectory (Task C):  {loss_dict['l_traj']:.4f}")
    print(f"   - OD Demand (Task D):   {loss_dict['l_od']:.4f}")
    print(f"   ----------------------------------------")
    print(f"   🎯 **Total Joint Loss ($\mathcal{{L}}_{{total}}$):** {total_loss.item():.4f}")

if __name__ == "__main__":
    run_tensor_verification()