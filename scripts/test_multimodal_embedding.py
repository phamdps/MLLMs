import torch
from src.models.multimodal import MultimodalUnifiedEmbedding # Assuming it's saved here or in src/models/

def run_multimodal_embedding_test():
    print("🚀 Initializing Multimodal Unified Embedding Test...")
    
    # Define batch size and dimensions matching our digital twin spec
    B = 2          # Batch size
    T_in = 12      # Historical time steps
    N = 207        # Number of sensors (Graph nodes)
    hidden_dim = 64 # Unified embedding dimension
    
    # Instantiate the embedding & fusion module
    model = MultimodalUnifiedEmbedding(num_sensors=N, hidden_dim=hidden_dim)
    model.eval()
    
    # --- 1. Create Synthetic Mock Tensors for Each Modality ---
    
    # A. Numerical Time Series: (Batch, Time_in, Sensors, Features)
    dummy_ts = torch.randn(B, T_in, N, 1)
    
    # B. Graph Topology / Node Features: (Batch, Sensors, Num_Nodes)
    dummy_graph = torch.randn(B, N, N)
    
    # C. Image Features (e.g., CNN / ViT backbone output): (Batch, 512)
    dummy_images = torch.randn(B, 512)
    
    # D. Text Features (e.g., Incident report embeddings): (Batch, 768)
    dummy_text = torch.randn(B, 768)
    
    print("\n📥 Input Modality Shapes:")
    print(f"   - Numerical Time Series: {dummy_ts.shape}")
    print(f"   - Graph Topology Matrix: {dummy_graph.shape}")
    print(f"   - Vision Features:       {dummy_images.shape}")
    print(f"   - Text Features:         {dummy_text.shape}")
    
    # --- 2. Forward Pass through Unified Embedding Space ---
    with torch.no_grad():
        unified_output = model(dummy_ts, dummy_graph, dummy_images, dummy_text)
        
    print("\n✅ Forward Pass Successful!")
    print(f"📤 Unified Spatiotemporal Representation Shape: {unified_output.shape}")
    print(f"   Expected: (Batch={B}, Sensors={N}, HiddenDim={hidden_dim})")
    
    # Assertions to guarantee correctness
    assert unified_output.shape == (B, N, hidden_dim), f"Shape mismatch! Got {unified_output.shape}"
    print("\n🎉 Test passed successfully! All four modalities are successfully mapped into the shared latent space.")

if __name__ == "__main__":
    run_multimodal_embedding_test()