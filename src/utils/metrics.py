import numpy as np
import json
from typing import Dict, Any, List, Tuple
from shapely.geometry import Polygon


class DigitalTwinMetrics:
    """
    Quantitative metrics library for multimodal Digital Twin models.
    Supports 3D bounding box IoU, trajectory errors (ADE/FDE), and LLM scoring.
    """

    @staticmethod
    def compute_2d_bev_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """
        Calculates Bird's-Eye-View (BEV) 2D Intersection over Union using Shapely.
        Box format: {"x": float, "y": float, "dx": float, "dy": float, "heading": float}
        """
        def get_polygon(b):
            x, y, dx, dy = b["x"], b["y"], b["dx"], b["dy"]
            # Oriented bounding box rectangle
            p = Polygon([
                (-dx / 2, -dy / 2),
                (-dx / 2, dy / 2),
                (dx / 2, dy / 2),
                (dx / 2, -dy / 2)
            ])
            # Rotate and translate
            from shapely.affinity import rotate, translate
            p = rotate(p, b.get("heading", 0), use_radians=False)
            p = translate(p, x, y)
            return p

        poly1 = get_polygon(box1)
        poly2 = get_polygon(box2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def compute_3d_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """Computes 3D Bounding Box IoU (BEV Area IoU * Height Overlap Ratio)."""
        bev_iou = DigitalTwinMetrics.compute_2d_bev_iou(box1, box2)
        
        # Calculate Z-axis overlap
        z1_min, z1_max = box1["z"] - box1["dz"] / 2, box1["z"] + box1["dz"] / 2
        z2_min, z2_max = box2["z"] - box2["dz"] / 2, box2["z"] + box2["dz"] / 2

        inter_z = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
        union_z = max(z1_max, z2_max) - min(z1_min, z2_min)

        z_overlap_ratio = inter_z / union_z if union_z > 0 else 0.0
        return bev_iou * z_overlap_ratio

    @staticmethod
    def compute_trajectory_errors(pred_traj: np.ndarray, gt_traj: np.ndarray) -> Dict[str, float]:
        """
        Computes Average Displacement Error (ADE) and Final Displacement Error (FDE).
        Args:
            pred_traj (np.ndarray): Predicted (T, 2) trajectory coordinates.
            gt_traj (np.ndarray): Ground truth (T, 2) trajectory coordinates.
        """
        distances = np.linalg.norm(pred_traj - gt_traj, axis=-1)
        ade = float(np.mean(distances))
        fde = float(distances[-1])
        return {"ADE": ade, "FDE": fde}

    @staticmethod
    def parse_json_response(response_text: str) -> Dict[str, Any]:
        """Extracts and validates JSON output from raw text model output."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback regex extraction if model output contains markdown code blocks
            import re
            match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError("Failed to extract valid JSON from model response.")