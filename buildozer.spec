[app]

  title = Zekra AI
  package.name = zekraai
  package.domain = org.zekra
  source.dir = .
  source.include_exts = py,png,jpg,kv,atlas,json,txt,onnx
  source.exclude_exts = spec
  source.exclude_dirs = tests,bin,venv,.git,__pycache__
  version = 2.0.0

  # الإصلاح الحرج: تثبيت numpy على 1.26.4 (آخر نسخة تتوافق مع NDK r25c)
  # numpy 2.x تستخدم std::unordered_map بطريقة غير متوافقة مع libc++ في NDK r25c
  requirements = python3,kivy,numpy==1.26.4,cython,android,jnius

  orientation = all
  android.services = ZekraService:service.py:foreground

  android.permissions = \
      CAMERA,\
      FOREGROUND_SERVICE,\
      FOREGROUND_SERVICE_MEDIA_PROJECTION,\
      RECORD_AUDIO,\
      READ_EXTERNAL_STORAGE,\
      WRITE_EXTERNAL_STORAGE,\
      SYSTEM_ALERT_WINDOW,\
      WAKE_LOCK,\
      RECEIVE_BOOT_COMPLETED

  android.api = 33
  android.minapi = 26

  # الإصلاح: NDK 25c هو ما يُحمَّل فعلاً في بيئة GitHub Actions
  android.ndk = 25c

  android.build_tools_version = 34.0.0
  android.private_storage = True
  android.allow_backup = False
  android.archs = arm64-v8a
  android.fullscreen = 1
  android.foreground_service = True
  android.logcat_filters = *:S python:D Zekra:V

  [buildozer]
  log_level = 2
  warn_on_root = 1
  