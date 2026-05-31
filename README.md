# On-Device AI Camera Processor

Professional offline Android application for real-time AI-powered object detection using Python, Kivy, and Buildozer.

## Features

- **Anti-Noise Filtering**: Gaussian blur eliminates sensor noise
- **Motion Compensation**: Farneback optical flow prevents false alerts during camera movement
- **Micro-Target Detection**: Isolates small moving objects (15x15+ pixels)
- **ONNX Inference**: Hardware-accelerated detection with CPU/GPU support
- **Strict Classification**: Alerts only for persons and cars with 60%+ confidence

## Quick Start

```bash
# 1. Download the YOLOv8 ONNX model
python download_model.py

# 2. Build the APK
buildozer android debug

# 3. Deploy to device
buildozer android deploy
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Kivy application with camera UI |
| `offline_engine.py` | ProfessionalVisionEngine class |
| `buildozer.spec` | Android build configuration |
| `requirements.txt` | Python dependencies |
| `download_model.py` | Model download script |

## Requirements

- Python 3.8+
- Buildozer
- YOLOv8 Nano ONNX model (downloaded via script)

## Architecture

```
Camera Frame -> Anti-Noise Filter -> Motion Compensation -> ONNX Inference -> Detection Overlay
```

## License

Educational use. Respect licenses of YOLOv8 (AGPL-3.0), ONNX Runtime (MIT), OpenCV (Apache 2.0), Kivy (MIT).
