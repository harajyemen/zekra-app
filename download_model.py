#!/usr/bin/env python3
"""
Zekra AI - YOLOv8 Model Downloader
====================================
يحمل نموذج YOLOv8n ويصدره بصيغة ONNX عبر ultralytics.
"""

import os
import sys

TARGET = "yolov8n.onnx"


def export_via_ultralytics():
    try:
        from ultralytics import YOLO
        print("[INFO] Exporting YOLOv8n via ultralytics...")
        model = YOLO("yolov8n.pt")
        export_path = model.export(
            format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False
        )
        if export_path and os.path.exists(str(export_path)):
            if str(export_path) != TARGET:
                import shutil
                shutil.copy(str(export_path), TARGET)
            print(f"[INFO] Model ready: {TARGET} ({os.path.getsize(TARGET)/1024/1024:.1f} MB)")
            return True
        print("[WARN] Export path not found")
        return False
    except ImportError:
        print("[WARN] ultralytics not installed")
        return False
    except Exception as e:
        print(f"[WARN] Export failed: {e}")
        return False


def download_direct():
    import urllib.request
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx",
    ]
    for url in urls:
        try:
            print(f"[INFO] Downloading: {url}")
            urllib.request.urlretrieve(url, TARGET)
            print(f"[INFO] Downloaded: {TARGET} ({os.path.getsize(TARGET)/1024/1024:.1f} MB)")
            return True
        except Exception as e:
            print(f"[WARN] Failed: {e}")
    return False


def main():
    print("[Zekra] ===== YOLOv8 Model Preparation =====")
    if os.path.exists(TARGET) and os.path.getsize(TARGET) > 1_000_000:
        print(f"[INFO] Model already exists ({os.path.getsize(TARGET)/1024/1024:.1f} MB)")
        sys.exit(0)
    if export_via_ultralytics():
        print("[Zekra] Model ready via ultralytics")
        sys.exit(0)
    if download_direct():
        print("[Zekra] Model ready via download")
        sys.exit(0)
    print("[INFO] No model found — app will run in demo mode")
    sys.exit(0)


if __name__ == "__main__":
    main()
