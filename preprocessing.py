#!/usr/bin/env python3
"""
Test script to verify CLAHE + gamma correction preprocessing.
Generates before/after comparison images.
"""

import cv2
import numpy as np
from pathlib import Path


def _apply_gamma_correction(frame: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction to brighten/darken frame."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(frame, table)


def _preprocess_frame(frame: np.ndarray, apply_gamma: bool = True) -> np.ndarray:
    """Preprocess frame with CLAHE + optional gamma correction."""
    # Convert BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply CLAHE on V channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    
    # Merge back to HSV and convert to BGR
    hsv_clahe = cv2.merge([h, s, v_clahe])
    frame_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    # Optionally apply gamma correction
    if apply_gamma:
        frame_clahe = _apply_gamma_correction(frame_clahe, gamma=1.15)
    
    return frame_clahe


def test_preprocessing(image_path: str, output_dir: str = "preprocessing_test_output"):
    """Test preprocessing on a single image."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"✅ Loaded image: {image_path}")
    print(f"   Shape: {frame.shape}")
    
    # Apply preprocessing
    preprocessed = _preprocess_frame(frame)
    
    # Save individual images
    original_path = output_path / "01_original.jpg"
    preprocessed_path = output_path / "02_preprocessed.jpg"
    
    cv2.imwrite(str(original_path), frame)
    cv2.imwrite(str(preprocessed_path), preprocessed)
    
    print(f"✅ Saved original: {original_path}")
    print(f"✅ Saved preprocessed: {preprocessed_path}")
    
    # Create side-by-side comparison
    h, w = frame.shape[:2]
    comparison = np.hstack([frame, preprocessed])
    comparison_path = output_path / "03_comparison_side_by_side.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"✅ Saved comparison: {comparison_path}")
    
    # Create difference heatmap (to see what changed)
    # Convert to grayscale for simpler difference
    orig_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prep_gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(orig_gray, prep_gray)
    diff_path = output_path / "04_difference_heatmap.jpg"
    cv2.imwrite(str(diff_path), diff)
    print(f"✅ Saved difference heatmap: {diff_path}")
    
    # Compute contrast metrics
    orig_contrast = orig_gray.std()
    prep_contrast = prep_gray.std()
    improvement = ((prep_contrast - orig_contrast) / orig_contrast) * 100
    
    print(f"\n📊 Contrast Analysis:")
    print(f"   Original contrast (std dev): {orig_contrast:.2f}")
    print(f"   Preprocessed contrast (std dev): {prep_contrast:.2f}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    print(f"\n📁 All outputs saved to: {output_path.absolute()}")


def test_from_camera(num_frames: int = 5):
    """Capture frames from camera and test preprocessing."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    output_path = Path("preprocessing_test_output")
    output_path.mkdir(exist_ok=True)
    
    print(f"📷 Capturing {num_frames} frames from camera...")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {i}")
            break
        
        preprocessed = _preprocess_frame(frame)
        comparison = np.hstack([frame, preprocessed])
        
        path = output_path / f"camera_frame_{i:02d}_comparison.jpg"
        cv2.imwrite(str(path), comparison)
        print(f"✅ Frame {i}: {path}")
    
    cap.release()
    print(f"\n✅ Camera test complete. Outputs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with provided image
        image_path = sys.argv[1]
        test_preprocessing(image_path)
    else:
        # Try to find a test image in common locations
        test_images = [
            "test.jpg",
            "sample.jpg",
            "frame.jpg",
            Path.cwd() / "test.jpg",
        ]
        
        found = False
        for img_path in test_images:
            if Path(img_path).exists():
                test_preprocessing(str(img_path))
                found = True
                break
        
        if not found:
            print("📸 No test image found. Options:")
            print("  1. Run: python test_preprocessing.py <path_to_image>")
            print("  2. Run: python test_preprocessing.py --camera  (captures from webcam)")
            print("\n   Example: python test_preprocessing.py sample.jpg")
            
            if len(sys.argv) > 1 and sys.argv[1] == "--camera":
                test_from_camera()
