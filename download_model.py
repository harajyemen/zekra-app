#!/usr/bin/env python3
"""
YOLOv8 ONNX Model Download Script - Automated CI Edition
========================================================
Downloads the pre-converted YOLOv8 Nano model in ONNX format 
automatically for offline deployment. Safe for GitHub Actions & CI builds.

Author: AI Development Team
Version: 1.2.0 (Silent Automate Build)
"""

import os
import sys
import urllib.request

def download_direct() -> bool:
    """تحميل مباشر وسريع للنموذج الجاهز والمصنع رسمياً لمنع استهلاك سيرفر البناء"""
    target_path = "yolov8n.onnx"
    
    # الرابط المباشر والأكثر استقراراً المعتمد من Ultralytics للـ ONNX
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
    
    # رابط احتياطي في حال حدوث أي مشكلة في الرابط الأول
    fallback_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"

    print("[INFO] Starting direct automated download for YOLOv8n ONNX model...")
    
    try:
        print(f"[INFO] Downloading from primary source: {url}")
        urllib.request.urlretrieve(url, target_path)
    except Exception as primary_error:
        print(f"[WARN] Primary source failed: {primary_error}")
        try:
            print(f"[INFO] Trying fallback source: {fallback_url}")
            urllib.request.urlretrieve(fallback_url, target_path)
        except Exception as fallback_error:
            print(f"[ERROR] All download locations failed: {fallback_error}")
            return False

    if os.path.exists(target_path):
        file_size_mb = os.path.getsize(target_path) / 1024 / 1024
        print(f"[SUCCESS] Model verified and saved: {os.path.abspath(target_path)}")
        print(f"[INFO] Model Size: {file_size_mb:.2f} MB")
        
        # التأكد من أن الملف سليم وحجمه طبيعي وليس فارغاً
        if file_size_mb > 1.0:
            return True
        else:
            print("[ERROR] Downloaded file is corrupted or too small.")
            return False
    else:
        print("[ERROR] Download completed but file was not registered on disk.")
        return False

def main():
    print("=" * 60)
    print("Automated YOLOv8 ONNX Model Downloader for Android Build")
    print("=" * 60)

    target_path = "yolov8n.onnx"
    
    # إذا كان الملف موجوداً مسبقاً، يتخطى التحميل مباشرة لمنع إضاعة الوقت في سيرفر البناء
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        print(f"[INFO] Model already exists natively: {os.path.abspath(target_path)}")
        print(f"[INFO] Size: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
        print("[INFO] Skipping download. Ready for bundling.")
        sys.exit(0)

    # تشغيل التحميل الأوتوماتيكي الصامت
    success = download_direct()

    print()
    if success:
        print("=" * 60)
        print("[SUCCESS] Automated model inclusion complete!")
        print("[INFO] Ready for Buildozer packaging workflow.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("[CRITICAL] Automated model preparation failed.")
        print("[FIX] Ensure network accessibility or embed the model manually in the repository.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
