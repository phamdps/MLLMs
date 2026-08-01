import os
import json
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
from PIL import Image


class DriveLMPromptBuilder:
    """
    Data loader and prompt builder for DriveLM / nuScenes dataset format.
    Transforms multi-camera images and graph-structured annotations into 
    standardized inputs for proprietary APIs and open-source MLLMs.
    """

    CAMERA_KEYS = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ]

    def __init__(self, dataroot: str, max_image_size: tuple = (800, 450)):
        """
        Args:
            dataroot (str): Path to root directory containing nuScenes images and DriveLM JSONs.
            max_image_size (tuple): Downsampling resolution (W, H) to manage context limits.
        """
        self.dataroot = dataroot
        self.max_image_size = max_image_size

    def load_annotation(self, json_path: str) -> Dict[str, Any]:
        """Loads a single DriveLM annotation file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_camera_images(self, frame_data: Dict[str, Any]) -> Dict[str, Image.Image]:
        """Loads and resizes multi-camera PIL Images."""
        images = {}
        images_info = frame_data.get("images", {})
        
        for cam_key in self.CAMERA_KEYS:
            if cam_key in images_info:
                rel_path = images_info[cam_key]
                full_path = os.path.join(self.dataroot, rel_path)
                if os.path.exists(full_path):
                    img = Image.open(full_path).convert("RGB")
                    if self.max_image_size:
                        img.thumbnail(self.max_image_size)
                    images[cam_key] = img
        return images

    def build_system_prompt(self) -> str:
        return (
            "You are an AI autonomous driving perception and planning system operating within a Digital Twin.\n"
            "You are provided with surround-view multi-camera images from six angles.\n"
            "Analyze the scene through sequential Graph-of-Thought reasoning:\n"
            "1. PERCEPTION: Identify dynamic objects, traffic lights, and road boundaries.\n"
            "2. PREDICTION: Forecast object trajectories and potential collision risks.\n"
            "3. PLANNING: Determine ego-vehicle maneuver and motion trajectory."
        )

    def build_user_prompt(self, frame_data: Dict[str, Any]) -> str:
        """Constructs text prompt from the DriveLM question graph."""
        qa_graph = frame_data.get("QA", {})
        prompt = "Review the multi-view images and answer the following driving graph queries:\n\n"

        for stage in ["perception", "prediction", "planning"]:
            if stage in qa_graph and qa_graph[stage]:
                prompt += f"=== {stage.upper()} STAGE ===\n"
                for i, item in enumerate(qa_graph[stage]):
                    q_text = item.get("question", item) if isinstance(item, dict) else str(item)
                    prompt += f"Q{i+1}: {q_text}\n"
                prompt += "\n"

        prompt += "Provide your answers in a valid JSON object matching the keys: 'perception', 'prediction', 'planning'."
        return prompt

    def image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """Encodes PIL image to Base64 string."""
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def format_openai_payload(self, json_path: str) -> Dict[str, Any]:
        """Generates payload for OpenAI GPT-4o style Vision API."""
        frame_data = self.load_annotation(json_path)
        images = self.load_camera_images(frame_data)
        
        content = [{"type": "text", "text": self.build_user_prompt(frame_data)}]
        
        for cam_name, img in images.items():
            b64_str = self.image_to_base64(img)
            content.append({
                "type": "text",
                "text": f"[{cam_name}]"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
            })

        return {
            "messages": [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": content}
            ]
        }