import os
import unittest
from src.dataloader.drivelm_builder import DriveLMPromptBuilder

class TestDriveLMPipeline(unittest.TestCase):

    def setUp(self):
        self.json_path = "./data/raw/DriveLM/sample_frame_001.json"
        self.builder = DriveLMPromptBuilder(dataroot="./data/raw/nuScenes")

    def test_json_loading(self):
        """Verify mock frame JSON parses without syntax errors."""
        data = self.builder.load_annotation(self.json_path)
        self.assertIn("images", data)
        self.assertIn("QA", data)
        self.assertEqual(len(data["images"]), 6)

    def test_prompt_generation(self):
        """Ensure system and user prompts format cleanly."""
        data = self.builder.load_annotation(self.json_path)
        user_prompt = self.builder.build_user_prompt(data)
        self.assertIn("PERCEPTION STAGE", user_prompt)
        self.assertIn("PLANNING STAGE", user_prompt)

    def test_missing_camera_graceful_handling(self):
        """Verify pipeline handles dropped/missing camera feeds gracefully."""
        data = self.builder.load_annotation(self.json_path)
        # Simulate camera hardware failure
        del data["images"]["CAM_BACK_RIGHT"]
        
        images = self.builder.load_camera_images(data)
        self.assertNotIn("CAM_BACK_RIGHT", images)
        print("Gracefully handled missing sensor feed!")

if __name__ == "__main__":
    unittest.main()

# 
# 1. Run Quantitative Metrics Unit Tests (3D IoU, Trajectories)
# python -m scripts.test_metrics

# 2. Run Data Pipeline & Sensor Integrity Tests
# python -m scripts.test_data_pipeline

# 3. Run LLM-as-a-Judge Evaluation Test (Requires API Key)
# python -m scripts.evaluate_reasoning