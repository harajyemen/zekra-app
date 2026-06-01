[app]

  # اسم التطبيق
  title = Zekra AI

  # اسم الحزمة
  package.name = zekraai

  # نطاق الحزمة
  package.domain = org.zekra

  # مجلد الكود المصدري
  source.dir = .

  # أنواع الملفات المضمنة في البناء
  source.include_exts = py,png,jpg,kv,atlas,json,txt,onnx

  # استثناء ملفات غير ضرورية
  source.exclude_exts = spec
  source.exclude_dirs = tests,bin,venv,.git,__pycache__

  # إصدار التطبيق
  version = 2.0.0

  # المتطلبات — فقط المتوافق مع أندرويد، بدون onnxruntime العامة
  requirements = python3,kivy,numpy,cython,android,jnius

  # الاتجاه
  orientation = all

  # خدمة الخلفية
  android.services = ZekraService:service.py:foreground

  # الصلاحيات الحتمية
  android.permissions = \
      CAMERA,\
      FOREGROUND_SERVICE,\
      FOREGROUND_SERVICE_MEDIA_PROJECTION,\
      RECORD_AUDIO,\
      READ_EXTERNAL_STORAGE,\
      WRITE_EXTERNAL_STORAGE,\
      SYSTEM_ALERT_WINDOW,\
      WAKE_LOCK,\
      RECEIVE_BOOT_COMPLETED,\
      REQUEST_INSTALL_PACKAGES

  # حقن صلاحية الـ Overlay الحرجة مباشرةً في AndroidManifest.xml
  android.add_activities = android.media.projection.MediaProjectionActivity

  # إصدار أندرويد المستهدف
  android.api = 33

  # أدنى إصدار مدعوم
  android.minapi = 26

  # إصدار NDK
  android.ndk = 25c

  # إصدار Build Tools
  android.build_tools_version = 33.0.0

  # التخزين الخاص
  android.private_storage = True

  # النسخ الاحتياطي
  android.allow_backup = False

  # المعمارية المستهدفة — arm64 فقط للهواتف الحديثة
  android.archs = arm64-v8a

  # ملء الشاشة الكاملة
  android.fullscreen = 1

  # فلاتر الـ logcat لتتبع الأخطاء بوضوح
  android.logcat_filters = *:S python:D Zekra:V

  # تفعيل التوقيع التلقائي للنسخة التجريبية
  android.release = False

  [buildozer]
  log_level = 2
  warn_on_root = 1
  