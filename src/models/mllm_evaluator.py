import os
import torch
from typing import List, Union
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

class MLLMEvaluator:
    """
    Inference harness for Qwen2-VL running locally on GPU.
    Uses 4-bit quantization to prevent CPU offloading and OOM errors during 6-camera processing.
    """
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct", 
        load_in_4bit: bool = True
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on GPU (4-bit: {load_in_4bit})...")

        self.processor = AutoProcessor.from_pretrained(model_name)

        if load_in_4bit and self.device == "cuda":
            # 4-bit quantization drastically reduces VRAM overhead (~5-6 GB VRAM footprint)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                attn_implementation="sdpa",
                device_map={"": 0}  # Forces ENTIRE model onto GPU 0 (no CPU offloading)
            ).eval()
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                attn_implementation="sdpa",
                device_map="auto"
            ).eval()

    def generate_response(
        self, 
        images: List[Union[str, Image.Image]], 
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2
    ) -> str:
        content = []
        for img in images:
            if isinstance(img, str):
                content.append({"type": "image", "image": f"file://{os.path.abspath(img)}"})
            elif isinstance(img, Image.Image):
                content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        return output_text