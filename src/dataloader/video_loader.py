"""
CCTV Video Loader for Visual Traffic Modality Integration
Handles video reading, frame sampling, and tensor formatting for Qwen2-VL.
"""

import os
import torch
from torch.utils.data import Dataset
try:
    from torchvision.io import read_video
except ImportError:
    read_video = None

class CCTVVideoDataset(Dataset):
    def __init__(self, video_dir: str, sample_fps: int = 1, max_frames: int = 8):
        """
        Args:
            video_dir (str): Path to folder containing intersection CCTV video mp4 files.
            sample_fps (int): Target frame rate to sample from videos.
            max_frames (int): Maximum frames to keep per sequence clip.
        """
        self.video_dir = video_dir
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.video_files = []
        
        if os.path.exists(video_dir):
            self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    def __len__(self):
        return len(self.video_files) if self.video_files else 1  # Fallback for empty sandbox

    def __getitem__(self, idx):
        if not self.video_files:
            # Return dummy tensor if no actual video files are present yet
            return {
                "video_tensor": torch.zeros((self.max_frames, 3, 224, 224), dtype=torch.float32),
                "meta": "synthetic_cctv_frame"
            }
            
        video_path = self.video_files[idx]
        # Using torchvision to read video frames
        v_frames, audio, info = read_video(video_path, pts_unit='sec')
        
        # Temporal subsampling to limit token length
        total_frames = v_frames.shape[0]
        indices = torch.linspace(0, total_frames - 1, steps=min(self.max_frames, total_frames)).long()
        sampled_frames = v_frames[indices] # Shape: (T, H, W, C)
        
        # Permute to (T, C, H, W) and normalize to [0, 1]
        sampled_frames = sampled_frames.permute(0, 3, 1, 2).float() / 255.0
        
        return {
            "video_tensor": sampled_frames,
            "meta": os.path.basename(video_path)
        }