"""Qwen2-VL Model Wrapper for Multimodal Traffic Prediction."""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class Qwen2VLTrafficWrapper:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
