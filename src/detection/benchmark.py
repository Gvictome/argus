"""
Benchmark YOLOv8.1 (8.1.x) performance on ARGUS.
Target: 30 FPS
"""

import time
import torch
import numpy as np
import cv2
from typing import List

try:
    from ultralytics import YOLO
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

def benchmark(model_name: str = "yolov8n.pt", iterations: int = 100):
    if not _AVAILABLE:
        print("[!] Error: ultralytics (YOLO) not installed.")
        return

    print(f"[*] Loading model: {model_name}...")
    model = YOLO(model_name)
    
    # Force evaluation mode
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy frame (1080p as per ARGUS settings)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    print(f"[*] Starting benchmark ({iterations} iterations)...")
    
    # Warmup
    for _ in range(5):
        model(dummy_frame, verbose=False)
    
    start_time = time.perf_counter()
    for i in range(iterations):
        model(dummy_frame, verbose=False)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    fps = 1 / avg_time
    
    print("-" * 30)
    print(f"RESULTS for {model_name}:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Inference: {avg_time*1000:.2f}ms")
    print(f"  Achieved FPS: {fps:.2f}")
    print("-" * 30)
    
    if fps >= 30:
        print("[✔] Target of 30 FPS reached!")
    else:
        print(f"[✘] Target of 30 FPS missed by {30 - fps:.2f} FPS.")
        print("    Recommendation: Export to 'hailo' HEF for NPU acceleration.")

if __name__ == "__main__":
    import sys
    model_to_test = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
    benchmark(model_to_test)
