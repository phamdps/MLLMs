import sys
from pathlib import Path

# Add project root directory to Python path dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
from PIL import Image
from src.evaluation.mllm_evaluator import MLLMEvaluator
from src.dataloader.drivelm_prompt_builder import DriveLMPromptBuilder

def main():
    print("=== Launching Local MLLM Evaluation Runner ===")

    # 1. Load DriveLM annotation frame
    json_path = PROJECT_ROOT / "data/raw/DriveLM/sample_frame_001.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file at {json_path}.")
        
    with open(json_path, "r") as f:
        frame_data = json.load(f)

    # 2. Extract and resize camera images to save VRAM
    nuscenes_root = PROJECT_ROOT / "data/raw/nuScenes"
    image_paths = [nuscenes_root / rel_path for rel_path in frame_data["images"].values()]

    processed_images = []
    for p in image_paths:
        if p.exists():
            img = Image.open(p).convert("RGB")
            # Resize surround views to 448x256 to fit comfortably in VRAM
            img = img.resize((448, 256))
            processed_images.append(img)
        else:
            print(f"[Warning] Frame missing at: {p}")

    if not processed_images:
        print("[Warning] No valid images found. Creating dummy frame...")
        dummy_img = Image.new('RGB', (448, 256), color=(73, 109, 137))
        processed_images = [dummy_img]
    else:
        print(f"Successfully loaded and resized {len(processed_images)} camera view(s).")

    # 3. Initialize Evaluator & Build Task Prompt
    evaluator = MLLMEvaluator(model_name="Qwen/Qwen2-VL-7B-Instruct")
    prompt_builder = DriveLMPromptBuilder()

    task_prompt = prompt_builder.build_graph_reasoning_prompt(
        question="What is the ego vehicle's safe action given the traffic light status and nearby obstacles?",
        output_format="JSON"
    )

    print(f"\n--- Submitting Multi-Camera Prompt to GPU ---")
    
    # 4. Run Local GPU Inference
    raw_response = evaluator.generate_response(
        images=processed_images,
        prompt=task_prompt,
        max_new_tokens=256
    )

    print("\n=== Raw Model Response ===")
    print(raw_response)

if __name__ == "__main__":
    main()