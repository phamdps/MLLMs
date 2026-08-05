import sys
from pathlib import Path
import json
from PIL import Image

# Add project root directory to Python path dynamically if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.mllm_evaluator import MLLMEvaluator
from src.dataloader.drivelm_prompt_builder import DriveLMPromptBuilder

def main():
    print("=== Launching Transportation Digital Twin MLLM Evaluator ===")

    # 1. Initialize the evaluator and prompt builder
    evaluator = MLLMEvaluator(
        model_name="Qwen/Qwen2-VL-7B-Instruct", 
        load_in_4bit=True
    )
    prompt_builder = DriveLMPromptBuilder()

    # 2. Load DriveLM annotation frame (similar to your working file)
    json_path = Path("data/raw/DriveLM/sample_frame_001.json")
    
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file at {json_path}.")
        
    with open(json_path, "r") as f:
        frame_data = json.load(f)

    # 3. Extract and resize camera images to save VRAM
    nuscenes_root = Path("data/raw/nuScenes")
    image_paths = [nuscenes_root / rel_path for rel_path in frame_data.get("images", {}).values()]

    processed_images = []
    for p in image_paths:
        if p.exists():
            img = Image.open(p).convert("RGB")
            # Resize views to 448x256 to fit comfortably in VRAM
            img = img.resize((448, 256))
            processed_images.append(img)
        else:
            print(f"⚠️ [Warning] Frame missing at: {p}")

    # Fallback if no images found
    if not processed_images:
        print("⚠️ No valid camera images found. Creating a placeholder frame...")
        dummy_img = Image.new('RGB', (448, 256), color=(73, 109, 137))
        processed_images = [dummy_img]
    else:
        print(f"✅ Successfully loaded and resized {len(processed_images)} camera view(s).")

    # 4. Construct a professional transportation digital twin prompt
    task_prompt = prompt_builder.build_graph_reasoning_prompt(
        question=(
            "You are an AI Transportation Digital Twin coordinator. "
            "Analyze the provided traffic camera feed(s). "
            "1. Estimate the current traffic density and congestion level (Low, Medium, High). "
            "2. Identify any visible bottlenecks, stalled vehicles, or safety hazards. "
            "3. Provide a brief recommendation for traffic light signal adjustments or lane management."
        ),
        output_format="JSON"
    )

    print(f"\n🚀 Running zero-shot inference on {len(processed_images)} camera view(s)...")
    
    # 5. Generate response from Qwen2-VL
    response = evaluator.generate_response(
        images=processed_images,
        prompt=task_prompt,
        max_new_tokens=512,
        temperature=0.2
    )

    print("\n" + "="*50)
    print("🚦 Qwen2-VL Transportation Digital Twin Analysis:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    main()