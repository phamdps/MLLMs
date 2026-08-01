import os
import json
from openai import OpenAI
from src.dataloader.drivelm_builder import DriveLMPromptBuilder

class LLMJudgeEvaluator:
    """Uses a judge model (GPT-4o) to score MLLM traffic explanations against rubrics."""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def score_reasoning(self, ground_truth: dict, model_prediction: dict) -> dict:
        judge_prompt = f"""
You are an expert Autonomous Driving Safety Inspector evaluating a model's perception and planning reasoning.

GROUND TRUTH ANNOTATION:
{json.dumps(ground_truth, indent=2)}

MODEL PREDICTED RESPONSE:
{json.dumps(model_prediction, indent=2)}

Evaluate the predicted response against the ground truth on a scale from 1 to 5 based on:
1. Spatial Perception Accuracy (Did it spot key pedestrians/vehicles?)
2. Collision Risk Realism (Is the risk prediction aligned with physics?)
3. Action Safety & Compliance (Is the braking/steering action safe and legal?)

Output strictly a JSON with key 'scores' (containing 'perception', 'risk', 'planning' scores 1-5) and 'rationale'.
"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

# Quick Execution Test
if __name__ == "__main__":
    evaluator = LLMJudgeEvaluator()
    
    # Load mock ground truth QA
    builder = DriveLMPromptBuilder(dataroot="./data/raw/nuScenes")
    gt_data = builder.load_annotation("./data/raw/DriveLM/sample_frame_001.json")["QA"]

    # Mock candidate model prediction
    mock_prediction = {
        "perception": "Observed pedestrian crossing from the left and dynamic vehicle on the right.",
        "prediction": "Pedestrian will cross the path in ~1.5 seconds, creating high collision risk.",
        "planning": "Apply moderate braking and stop before the crosswalk line."
    }

    if os.getenv("OPENAI_API_KEY"):
        result = evaluator.score_reasoning(gt_data, mock_prediction)
        print("--- LLM-as-a-Judge Evaluation Output ---")
        print(json.dumps(result, indent=2))
    else:
        print("Skipping LLM-as-a-Judge API call: OPENAI_API_KEY not set in environment.")