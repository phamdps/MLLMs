import numpy as np
from src.utils.metrics import DigitalTwinMetrics

def test_spatial_metrics():
    print("=== TEST 1: 3D Bounding Box IoU ===")
    
    # Target Ground Truth Vehicle
    gt_box = {"x": 4.0, "y": 18.0, "z": 0.0, "dx": 2.0, "dy": 4.5, "dz": 1.5, "heading": 0.0}
    
    # Model Predictions: Perfect match vs. Shifted prediction
    pred_perfect = {"x": 4.0, "y": 18.0, "z": 0.0, "dx": 2.0, "dy": 4.5, "dz": 1.5, "heading": 0.0}
    pred_shifted = {"x": 4.5, "y": 18.5, "z": 0.2, "dx": 2.0, "dy": 4.5, "dz": 1.5, "heading": 5.0}

    iou_perfect = DigitalTwinMetrics.compute_3d_iou(gt_box, pred_perfect)
    iou_shifted = DigitalTwinMetrics.compute_3d_iou(gt_box, pred_shifted)

    print(f"Perfect Overlap 3D IoU: {iou_perfect:.4f} (Expected: 1.0000)")
    print(f"Shifted Overlap 3D IoU: {iou_shifted:.4f} (Expected: ~0.50-0.70)")
    assert iou_perfect == 1.0, "3D IoU computation failed for identical boxes!"

def test_trajectory_metrics():
    print("\n=== TEST 2: Trajectory Error (ADE / FDE) ===")
    
    # Simulated 3-second trajectory (X, Y positions at 1-second intervals)
    gt_trajectory = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0]])
    pred_trajectory = np.array([[0.2, 0.1], [0.5, 5.2], [1.0, 10.8]])

    errors = DigitalTwinMetrics.compute_trajectory_errors(pred_trajectory, gt_trajectory)
    print(f"Average Displacement Error (ADE): {errors['ADE']:.3f} meters")
    print(f"Final Displacement Error (FDE):   {errors['FDE']:.3f} meters")

if __name__ == "__main__":
    test_spatial_metrics()
    test_trajectory_metrics()