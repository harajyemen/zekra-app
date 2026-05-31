[app]

# (str) Title of your application
title = On-Device AI Camera Processor

# (str) Package name
package.name = aicameraprocessor

# (str) Package domain (needed for android/ios packaging)
package.domain = org.offline

# (str) Source code where the main entry point is located
source.dir = .

# (list) Source files to include (empty for all)
source.include_exts = py,png,jpg,kv,atlas,onnx,json,txt

# (list) List of files to exclude
source.exclude_exts = spec

# (str) Application versioning
version = 1.0.0

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,numpy,opencv-python-headless,onnxruntime

# (str) Supported orientation (landscape, portrait or all)
orientation = all

# (list) Permissions
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,VIBRATE,WAKE_LOCK

# (int) Target Android API, should be greater or equal to 21
android.api = 31

# (int) Minimum API your APK will support
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (bool) Use --private data storage
android.private_storage = True

# (bool) enables Android auto backup feature (Android API >=23)
android.allow_backup = True

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

# (bool) Enables full filesystem access for the app
android.fullscreen = True

# (bool) If True, then the app will be compiled with the release flag
android.release = True

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1
