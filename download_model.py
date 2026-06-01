#!/usr/bin/env python3
  """
  Zekra AI - YOLOv8 Model Downloader
  ====================================
  يحمّل نموذج YOLOv8n ويصدّره بصيغة ONNX عبر مكتبة ultralytics الرسمية.
  هذا أكثر موثوقية من التحميل المباشر للملف.

  Author: Zekra AI Team
  Version: 2.1.0
  """

  import os
  import sys

  TARGET = "yolov8n.onnx"

  def export_via_ultralytics() -> bool:
      """
      استخدام ultralytics لتحميل YOLOv8n.pt وتصديره كـ ONNX.
      هذا هو الأسلوب الرسمي والأكثر استقراراً.
      """
      try:
          from ultralytics import YOLO
          print("[INFO] ultralytics available — downloading and exporting YOLOv8n...")
          
          # تحميل النموذج (يحمّل .pt تلقائياً إذا لم يكن موجوداً)
          model = YOLO("yolov8n.pt")
          
          # التصدير بصيغة ONNX
          export_path = model.export(
              format="onnx",
              imgsz=640,
              opset=12,
              simplify=True,
              dynamic=False
          )
          
          # نسخ الملف المُصدَّر إلى المسار المطلوب
          if export_path and os.path.exists(export_path):
              if str(export_path) != TARGET:
                  import shutil
                  shutil.copy(str(export_path), TARGET)
              print(f"[INFO] Model exported successfully: {TARGET}")
              size_mb = os.path.getsize(TARGET) / 1024 / 1024
              print(f"[INFO] File size: {size_mb:.1f} MB")
              return True
          else:
              print(f"[WARN] Export path not found: {export_path}")
              return False
              
      except ImportError:
          print("[WARN] ultralytics not installed — trying direct download")
          return False
      except Exception as e:
          print(f"[WARN] ultralytics export failed: {e}")
          return False


  def download_direct() -> bool:
      """
      تحميل مباشر من مصادر GitHub الرسمية (بديل احتياطي).
      """
      import urllib.request
      
      # أحدث روابط مجربة من Ultralytics Releases
      urls = [
          "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
          "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
          "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx",
          "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
      ]
      
      for url in urls:
          try:
              print(f"[INFO] Trying: {url}")
              urllib.request.urlretrieve(url, TARGET)
              size_mb = os.path.getsize(TARGET) / 1024 / 1024
              print(f"[INFO] Downloaded: {TARGET} ({size_mb:.1f} MB)")
              return True
          except Exception as e:
              print(f"[WARN] Failed: {e}")
      
      return False


  def main():
      print("[Zekra] ========== YOLOv8 Model Preparation ==========")
      
      # إذا كان الملف موجوداً، لا نعيد تحميله
      if os.path.exists(TARGET) and os.path.getsize(TARGET) > 1_000_000:
          size_mb = os.path.getsize(TARGET) / 1024 / 1024
          print(f"[INFO] Model already exists: {TARGET} ({size_mb:.1f} MB)")
          sys.exit(0)
      
      # محاولة 1: ultralytics (الأفضل)
      if export_via_ultralytics():
          print("[Zekra] Model ready via ultralytics export")
          sys.exit(0)
      
      # محاولة 2: تحميل مباشر
      if download_direct():
          print("[Zekra] Model ready via direct download")
          sys.exit(0)
      
      # فشل كلا الأسلوبين
      print("[CRITICAL] Could not obtain YOLOv8 ONNX model.")
      print("[INFO] The app will run in demo mode without AI detection.")
      print("[INFO] To fix: manually place 'yolov8n.onnx' in the project root.")
      # لا نفشل البناء — التطبيق يعمل بنمط demo بدون النموذج
      sys.exit(0)


  if __name__ == '__main__':
      main()
  