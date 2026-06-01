[app]

# (str) Title of your application
title = On-Device AI Camera Processor

# (str) Package name
package.name = aicameraprocessor

# (str) Package domain
package.domain = org.offline

# (str) Source code directory where the main.py lives
source.dir = .

# (list) Source files to include
source.include_exts = py,png,jpg,kv,atlas,json,txt

# (list) Exclude onnx model (downloaded at runtime)
source.exclude_exts = spec,onnx

# (str) Application version
version = 1.0.0

# (list) Application requirements
# تم الحفاظ على معاييرك وإضافة jnius و android لربط أذونات البث
requirements = python3,kivy,numpy,opencv,onnxruntime,jnius,android

# (str) Supported orientation
orientation = all

# (list) Permissions of your application
# تم دمج صلاحياتك القديمة مع صلاحيات البث المباشر والظهور فوق ببجي الحتمية
android.permissions = CAMERA, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, VIBRATE, WAKE_LOCK, FOREGROUND_SERVICE, FOREGROUND_SERVICE_MEDIA_PROJECTION, SYSTEM_ALERT_WINDOW

# (str) Extra Manifest XML to inject into AndroidManifest.xml (Crucial for PUBG Overlay)
# هذا السطر يجبر نظام الأندرويد على إعطاء التطبيق ميزة الرسم فوق الألعاب
android.manifest.permissions = android.permission.SYSTEM_ALERT_WINDOW

# (list) Services to declare
# ربط الخدمة الخلفية لضمان استمرار البث المباشر والذكاء الاصطناعي أثناء اللعب
android.services = myservice:service.py

# (int) Target Android API
android.api = 31

# (int) Minimum API
android.minapi = 24

# (str) Android NDK version — r27c is the minimum needed for numpy 2.x
android.ndk = 27c

# (str) Build-tools version — must match what we pre-install in CI
android.build_tools_version = 34.0.0

# (bool) Private data storage
android.private_storage = True

# (bool) Android auto backup
android.allow_backup = True

# (list) Build for arm64-v8a only (faster; covers all modern devices like S21 Ultra)
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
