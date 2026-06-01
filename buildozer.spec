[app]

# (str) Title of your application
title = On-Device AI Camera Processor

# (str) Package name
package.name = aicameraprocessor

# (str) Package domain
package.domain = org.offline

# (str) Source code directory where the main.py lives
source.dir = .

# (list) Source files to include (تأكيد تضمين ملف الـ onnx هنا)
source.include_exts = py,png,jpg,kv,atlas,json,txt,onnx

# (list) Exclude specific extensions from the build
source.exclude_exts = spec

# (str) Application version
version = 1.0.0

# (list) Application requirements
# تصحيح شامل: الاعتماد على النسخ المتوافقة مع أندرويد وإزالة onnxruntime المسببة للكراش المباشر
requirements = python3,kivy,numpy,opencv-python,jnius,android

# (str) Supported orientation
orientation = all

# (list) Permissions of your application
android.permissions = CAMERA, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, VIBRATE, WAKE_LOCK, FOREGROUND_SERVICE, FOREGROUND_SERVICE_MEDIA_PROJECTION, SYSTEM_ALERT_WINDOW

# (str) Extra Manifest XML to inject into AndroidManifest.xml (Crucial for PUBG Overlay)
android.manifest.permissions = android.permission.SYSTEM_ALERT_WINDOW

# (list) Services to declare
android.services = myservice:service.py

# (int) Target Android API
android.api = 31

# (int) Minimum API
android.minapi = 24

# (str) Android NDK version
android.ndk = 27c

# (str) Build-tools version
android.build_tools_version = 34.0.0

# (bool) Private data storage
android.private_storage = True

# (bool) Android auto backup
android.allow_backup = True

# (list) Build for arm64-v8a only (faster)
android.archs = arm64-v8a

# (bool) Full screen
android.fullscreen = True

# (bool) Allow service to be run in foreground
android.foreground_service = True

# (bool) Debug build - no signing keystore needed
android.release = False

# (str) android logcat filters to use
android.logcat_filters = *:S python:D

[buildozer]

# (int) Log level (2 = debug with full output)
log_level = 2

# (int) Warn if run as root
warn_on_root = 1
