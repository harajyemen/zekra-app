[app]

# Title of your application
title = On-Device AI Camera Processor

# Package name
package.name = aicameraprocessor

# Package domain
package.domain = org.offline

# Source code directory
source.dir = .

# Source files to include
source.include_exts = py,png,jpg,kv,atlas,json,txt

# Exclude onnx model (downloaded at runtime)
source.exclude_exts = spec,onnx

# Application version
version = 1.0.0

# Requirements:
#   opencv     = p4a recipe (NOT opencv-python-headless, desktop-only)
#   numpy      = no version pin (p4a recipe handles version internally)
#   onnxruntime = installed via pip wheel
requirements = python3,kivy,numpy,opencv,onnxruntime

# Supported orientation
orientation = all

# Android permissions
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,VIBRATE,WAKE_LOCK

# Target Android API
android.api = 31

# Minimum API
android.minapi = 24

# Android NDK version
android.ndk = 25b

# Private data storage
android.private_storage = True

# Android auto backup
android.allow_backup = True

# Build for arm64-v8a only (faster; covers all modern devices)
android.archs = arm64-v8a

# Full screen
android.fullscreen = True

# Debug build - no signing keystore needed
android.release = False

[buildozer]

# Log level (2 = debug with full output)
log_level = 2

# Warn if run as root
warn_on_root = 1
