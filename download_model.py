#!/usr/bin/env python3
"""
YOLOv8 ONNX Model Download Script
=================================
Downloads and exports YOLOv8 Nano model in ONNX format for offline inference.

Usage:
    python download_model.py
"""

import os
import sys
import urllib.request
import subprocess


def download_with_ultralytics():
    """Download model using ultralytics library (recommended)."""
    try:
        print("[INFO] Attempting to download using ultralytics...")

        try:
            import ultralytics
        except ImportError:
            print("[INFO] Installing ultralytics package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            import ultralytics

        from ultralytics import YOLO

        print("[INFO] Downloading YOLOv8 Nano model...")
        model = YOLO("yolov8n.pt")

        print("[INFO] Exporting to ONNX format...")
        model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)

        export_path = "yolov8n.onnx"
        if os.path.exists(export_path):
            print(f"[SUCCESS] Model saved: {os.path.abspath(export_path)}")
            print(f"[INFO] Model size: {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            print("[ERROR] Export completed but file not found")
            return False

    except Exception as e:
        print(f"[ERROR] Ultralytics method failed: {e}")
        return False


def download_direct():
    """Download pre-converted ONNX model directly."""
    print("[INFO] Attempting direct download...")

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
    output_path = "yolov8n.onnx"

    try:
        print(f"[INFO] Downloading from: {url}")
        urllib.request.urlretrieve(url, output_path)

        if os.path.exists(output_path):
            print(f"[SUCCESS] Model saved: {os.path.abspath(output_path)}")
            print(f"[INFO] Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            print("[ERROR] Download completed but file not found")
            return False

    except Exception as e:
        print(f"[ERROR] Direct download failed: {e}")
        return False


def main():
    """Main download routine."""
    print("=" * 60)
    print("YOLOv8 ONNX Model Downloader")
    print("=" * 60)
    print()

    target_path = "yolov8n.onnx"
    if os.path.exists(target_path):
        print(f"[INFO] Model already exists: {os.path.abspath(target_path)}")
        print(f"[INFO] Size: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
        response = input("[PROMPT] Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("[INFO] Keeping existing model")
            return

    print()
    print("Select download method:")
    print("  1. Ultralytics (recommended)")
    print("  2. Direct download (pre-converted)")
    print("  0. Exit")
    print()

    choice = input("[PROMPT] Enter choice (1/2/0): ").strip()

    success = False

    if choice == '1':
        success = download_with_ultralytics()
    elif choice == '2':
        success = download_direct()
    elif choice == '0':
        print("[INFO] Cancelled")
        return
    else:
        print("[WARN] Invalid choice, trying all methods...")
        success = download_with_ultralytics() or download_direct()

    print()
    if success:
        print("=" * 60)
        print("[SUCCESS] Model download complete!")
        print("[INFO] You can now build the APK with: buildozer android debug")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[ERROR] Model download failed!")
        print("[INFO] Please download manually from:")
        print("[INFO] https://github.com/ultralytics/assets/releases")
        print("=" * 60)


if __name__ == "__main__":
    main()
