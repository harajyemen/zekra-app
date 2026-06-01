"""
  Zekra AI - التطبيق الرئيسي
  ============================
  تطبيق أندرويد للمساعدة البصرية الذكية للاعبين.
  يلتقط بث الشاشة الحي ويرسم مربعات تتبع فوسفورية فوق الألعاب.

  Author: Zekra AI Team
  Version: 2.0.0
  """

  import os
  import sys
  import time
  import threading
  from typing import Optional, List, Tuple

  # ===== اكتشاف المنصة =====
  from kivy.utils import platform
  IS_ANDROID = platform == 'android'

  # ===== استيرادات Kivy الأساسية =====
  from kivy.app import App
  from kivy.clock import Clock
  from kivy.core.window import Window
  from kivy.graphics import Color, Line, Rectangle, RoundedRectangle
  from kivy.graphics.texture import Texture
  from kivy.lang import Builder
  from kivy.properties import BooleanProperty, NumericProperty, StringProperty, ListProperty
  from kivy.uix.floatlayout import FloatLayout
  from kivy.uix.label import Label
  from kivy.uix.button import Button
  from kivy.uix.boxlayout import BoxLayout
  from kivy.uix.widget import Widget

  # ===== استيراد محرك الرؤية بأمان =====
  try:
      from offline_engine import ProfessionalVisionEngine, TrackedObject
      ENGINE_AVAILABLE = True
  except Exception as _eng_err:
      ENGINE_AVAILABLE = False
      TrackedObject = None
      print(f"[Zekra] Engine deferred: {_eng_err}")

  # ===== استيراد مكتبات OpenCV بأمان =====
  try:
      import numpy as np
      NUMPY_AVAILABLE = True
  except ImportError:
      NUMPY_AVAILABLE = False
      np = None

  try:
      import cv2
      CV2_AVAILABLE = True
  except ImportError:
      CV2_AVAILABLE = False

  # ===== استيرادات أندرويد (فقط على الجهاز) =====
  ANDROID_PERM_AVAILABLE = False
  MEDIA_PROJECTION_AVAILABLE = False

  if IS_ANDROID:
      try:
          from android.permissions import (
              request_permissions, check_permission,
              Permission
          )
          from android import mActivity
          ANDROID_PERM_AVAILABLE = True
          print("[Zekra] Android permissions module loaded")
      except Exception as e:
          print(f"[Zekra] android.permissions not available: {e}")

      try:
          from jnius import autoclass, cast
          _Context      = autoclass('android.content.Context')
          _MPManager    = autoclass('android.media.projection.MediaProjectionManager')
          _ImageReader  = autoclass('android.media.ImageReader')
          _PixelFormat  = autoclass('android.graphics.PixelFormat')
          _Surface      = autoclass('android.view.Surface')
          _Handler      = autoclass('android.os.Handler')
          _Looper       = autoclass('android.os.Looper')
          MEDIA_PROJECTION_AVAILABLE = True
          print("[Zekra] MediaProjection APIs loaded")
      except Exception as e:
          print(f"[Zekra] MediaProjection not available: {e}")


  # ===== ثوابت عالمية =====
  REQUEST_CODE_SCREEN = 1001
  REQUIRED_PERMISSIONS = [
      "android.permission.CAMERA",
      "android.permission.READ_EXTERNAL_STORAGE",
      "android.permission.WRITE_EXTERNAL_STORAGE",
  ]

  # ألوان فوسفورية لكل فئة
  CLASS_COLORS = {
      'person':     (0.0,  1.0,  1.0,  1.0),   # سماوي فوسفوري
      'car':        (1.0,  0.84, 0.0,  1.0),   # ذهبي فوسفوري
      'motorcycle': (1.0,  0.27, 0.0,  1.0),   # برتقالي
      'bicycle':    (0.56, 1.0,  0.0,  1.0),   # أخضر ليموني
      'truck':      (1.0,  0.41, 0.71, 1.0),   # وردي
      'bus':        (0.94, 1.0,  0.0,  1.0),   # أصفر
      'default':    (1.0,  1.0,  1.0,  1.0),   # أبيض
  }

  PREDICTED_COLOR = (0.5, 0.5, 1.0, 0.6)   # أزرق شفاف للمواضع المتوقعة


  # ===== واجهة المستخدم بلغة KV =====
  KV = """
  #:kivy 2.2.0

  <ZekraOverlay>:
      canvas.before:
          Color:
              rgba: 0, 0, 0, 0

  <StatusBar>:
      size_hint_y: None
      height: dp(48)
      padding: dp(8), dp(4)
      spacing: dp(8)
      canvas.before:
          Color:
              rgba: 0, 0, 0, 0.65
          RoundedRectangle:
              pos: self.pos
              size: self.size
              radius: [dp(8)]

      Label:
          id: lbl_status
          text: root.status_text
          color: 0, 0.9, 1, 1
          font_size: dp(13)
          halign: 'left'
          text_size: self.size
          size_hint_x: 0.7

      Label:
          id: lbl_fps
          text: root.fps_text
          color: 0.56, 1, 0, 1
          font_size: dp(13)
          halign: 'right'
          text_size: self.size
          size_hint_x: 0.3

  <PermissionScreen>:
      orientation: 'vertical'
      padding: dp(32)
      spacing: dp(16)
      canvas.before:
          Color:
              rgba: 0.05, 0.05, 0.12, 0.95
          Rectangle:
              pos: self.pos
              size: self.size

      Label:
          text: 'ZEKRA AI'
          font_size: dp(36)
          color: 0, 0.9, 1, 1
          bold: True
          size_hint_y: None
          height: dp(60)

      Label:
          text: 'المساعد البصري الذكي للاعبين'
          font_size: dp(16)
          color: 0.8, 0.8, 0.9, 1
          size_hint_y: None
          height: dp(32)

      Widget:
          size_hint_y: 0.1

      Label:
          id: perm_status
          text: root.perm_message
          font_size: dp(14)
          color: 1, 0.9, 0.3, 1
          halign: 'center'
          text_size: self.width, None
          size_hint_y: None
          height: dp(80)

      Button:
          id: btn_grant
          text: 'منح الصلاحيات'
          font_size: dp(16)
          size_hint_y: None
          height: dp(52)
          background_color: 0, 0.7, 1, 1
          bold: True
          on_press: root.request_permissions()

      Button:
          id: btn_overlay
          text: 'تفعيل طبقة العرض العلوية'
          font_size: dp(14)
          size_hint_y: None
          height: dp(48)
          background_color: 0.56, 1, 0, 0.9
          on_press: root.request_overlay_permission()

      Widget:
          size_hint_y: 0.2

  <ZekraMainLayout>:
      canvas.before:
          Color:
              rgba: 0, 0, 0, 0.01
          Rectangle:
              pos: self.pos
              size: self.size
  """
  Builder.load_string(KV)


  # ===== ويدجت رسم مربعات التتبع =====

  class ZekraOverlay(Widget):
      """
      طبقة شفافة ترسم فوقها مربعات التتبع الفوسفورية.
      تُرسم بالكامل باستخدام Kivy Canvas Instructions.
      """

      def __init__(self, **kwargs):
          super().__init__(**kwargs)
          self._tracked_objects: List = []
          self._lock = threading.Lock()

      def update_tracked(self, objects: List):
          """تحديث قائمة الأجسام المتتبعة (آمن من threads متعددة)."""
          with self._lock:
              self._tracked_objects = list(objects)
          self.canvas.ask_update()

      def on_size(self, *args):
          self._redraw()

      def on_pos(self, *args):
          self._redraw()

      def _redraw(self):
          self.canvas.after.clear()
          with self._lock:
              objects = list(self._tracked_objects)

          if not objects:
              return

          win_w = self.width
          win_h = self.height

          with self.canvas.after:
              for obj in objects:
                  x1, y1, x2, y2 = obj.bbox
                  color = PREDICTED_COLOR if obj.is_predicted else CLASS_COLORS.get(
                      obj.class_name, CLASS_COLORS['default'])

                  # ===== رسم المربع المحيط =====
                  # تحويل إحداثيات الإطار إلى إحداثيات الشاشة
                  # نفترض أن الإطار ذو نفس أبعاد النافذة
                  bx = x1
                  by = win_h - y2   # عكس محور Y (Kivy يبدأ من الأسفل)
                  bw = x2 - x1
                  bh = y2 - y1

                  # خط خارجي سميك
                  Color(rgba=color)
                  Line(
                      rectangle=(bx, by, bw, bh),
                      width=2.5
                  )

                  # زوايا مميزة (L-shape corners) لمظهر احترافي
                  corner = min(bw, bh) * 0.25
                  # زاوية يسار-أسفل
                  Line(points=[bx, by+corner, bx, by, bx+corner, by], width=3.5)
                  # زاوية يمين-أسفل
                  Line(points=[bx+bw-corner, by, bx+bw, by, bx+bw, by+corner], width=3.5)
                  # زاوية يسار-أعلى
                  Line(points=[bx, by+bh-corner, bx, by+bh, bx+corner, by+bh], width=3.5)
                  # زاوية يمين-أعلى
                  Line(points=[bx+bw-corner, by+bh, bx+bw, by+bh, bx+bw, by+bh-corner], width=3.5)

                  # ===== تسمية الجسم =====
                  # (Kivy لا يدعم رسم نص مباشرة على Canvas — نستخدم Label خارجي)


  # ===== شريط الحالة =====

  class StatusBar(BoxLayout):
      status_text = StringProperty("جارٍ التهيئة…")
      fps_text    = StringProperty("0.0 FPS")


  # ===== شاشة طلب الصلاحيات =====

  class PermissionScreen(BoxLayout):
      perm_message = StringProperty("يتطلب التطبيق صلاحيات لبدء التشغيل.")

      def request_permissions(self):
          """طلب صلاحيات الكاميرا والتخزين."""
          if ANDROID_PERM_AVAILABLE:
              request_permissions(
                  [Permission.CAMERA,
                   Permission.READ_EXTERNAL_STORAGE,
                   Permission.WRITE_EXTERNAL_STORAGE],
                  self._on_permissions_result
              )
              self.perm_message = "⏳ بانتظار الصلاحيات…"
          else:
              self.perm_message = "✓ وضع سطح المكتب — جاهز"
              app = App.get_running_app()
              if app:
                  Clock.schedule_once(lambda dt: app.on_permissions_granted(), 0.5)

      def _on_permissions_result(self, permissions, grant_results):
          """استقبال نتيجة طلب الصلاحيات."""
          all_granted = all(grant_results)
          if all_granted:
              self.perm_message = "✓ تم منح جميع الصلاحيات"
              app = App.get_running_app()
              if app:
                  Clock.schedule_once(lambda dt: app.on_permissions_granted(), 0.3)
          else:
              denied = [p for p, g in zip(permissions, grant_results) if not g]
              self.perm_message = f"✗ يرجى منح الصلاحيات يدوياً:\n{', '.join(denied)}"

      def request_overlay_permission(self):
          """فتح إعدادات أندرويد لمنح صلاحية العرض فوق التطبيقات."""
          if not IS_ANDROID:
              return
          try:
              from jnius import autoclass
              Intent    = autoclass('android.content.Intent')
              Settings  = autoclass('android.provider.Settings')
              Uri       = autoclass('android.net.Uri')
              from android import mActivity

              intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION)
              intent.setData(Uri.parse(f"package:{mActivity.getPackageName()}"))
              mActivity.startActivity(intent)
              self.perm_message = "⚙️ افتح وفعّل 'العرض فوق التطبيقات'"
          except Exception as e:
              self.perm_message = f"تعذّر فتح الإعدادات: {e}"


  # ===== واجهة رئيسية =====

  class ZekraMainLayout(FloatLayout):
      pass


  # ===== منطق MediaProjection لالتقاط الشاشة =====

  class ScreenCaptureManager:
      """
      يدير التقاط بث الشاشة الحي عبر MediaProjection API.
      يعمل فقط على أندرويد مع الصلاحيات الكاملة.
      """

      def __init__(self, on_frame_callback):
          self._callback      = on_frame_callback
          self._projection    = None
          self._virtual_disp  = None
          self._image_reader  = None
          self._running       = False
          self._thread        = None

      def request_capture(self):
          """بدء طلب صلاحية تصوير الشاشة من المستخدم."""
          if not MEDIA_PROJECTION_AVAILABLE:
              print("[ScreenCapture] Not on Android — using demo mode")
              self._start_demo_mode()
              return

          try:
              pm = mActivity.getSystemService(_Context.MEDIA_PROJECTION_SERVICE)
              intent = pm.createScreenCaptureIntent()

              # تسجيل listener لنتيجة الـ Intent
              def _on_activity_result(request_code, result_code, data):
                  if request_code == REQUEST_CODE_SCREEN and result_code == -1:  # RESULT_OK
                      self._on_projection_granted(pm, result_code, data)
                  else:
                      print("[ScreenCapture] User denied screen capture")

              mActivity.setActivityResultListener(_on_activity_result)
              mActivity.startActivityForResult(intent, REQUEST_CODE_SCREEN)
              print("[ScreenCapture] Screen capture dialog shown")

          except Exception as e:
              print(f"[ScreenCapture] request_capture error: {e}")
              self._start_demo_mode()

      def _on_projection_granted(self, pm, result_code, data):
          """استقبال إذن تصوير الشاشة وبدء VirtualDisplay."""
          try:
              from jnius import autoclass
              DisplayMetrics = autoclass('android.util.DisplayMetrics')
              metrics = DisplayMetrics()
              mActivity.getWindowManager().getDefaultDisplay().getMetrics(metrics)

              screen_w = metrics.widthPixels
              screen_h = metrics.heightPixels
              screen_dpi = metrics.densityDpi

              # إنشاء ImageReader لاستقبال الإطارات
              self._image_reader = _ImageReader.newInstance(
                  screen_w, screen_h,
                  _PixelFormat.RGBA_8888,
                  3   # ثلاثة إطارات في الـ buffer
              )

              # الحصول على MediaProjection
              self._projection = pm.getMediaProjection(result_code, data)

              # إنشاء VirtualDisplay
              self._virtual_disp = self._projection.createVirtualDisplay(
                  "ZekraCapture",
                  screen_w, screen_h, screen_dpi,
                  0x10,   # VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR
                  self._image_reader.getSurface(),
                  None, None
              )

              # بدء thread قراءة الإطارات
              self._running = True
              self._thread  = threading.Thread(
                  target=self._frame_reader_loop,
                  args=(screen_w, screen_h),
                  name="ZekraFrameReader",
                  daemon=True
              )
              self._thread.start()
              print(f"[ScreenCapture] VirtualDisplay started @ {screen_w}x{screen_h}")

          except Exception as e:
              print(f"[ScreenCapture] _on_projection_granted error: {e}")
              self._start_demo_mode()

      def _frame_reader_loop(self, screen_w: int, screen_h: int):
          """
          حلقة قراءة الإطارات من ImageReader بسرعة 60 FPS.
          تحوّل كل إطار إلى numpy array وتستدعي callback المعالجة.
          """
          print("[ScreenCapture] Frame reader loop started (target: 60 FPS)")
          frame_interval = 1.0 / 60.0   # 60 FPS

          while self._running:
              t0 = time.time()
              try:
                  image = self._image_reader.acquireLatestImage()
                  if image is not None:
                      # استخراج البيانات من الـ Image
                      planes  = image.getPlanes()
                      buffer  = planes[0].getBuffer()
                      row_stride = planes[0].getRowStride()

                      if NUMPY_AVAILABLE:
                          # تحويل buffer إلى numpy array
                          raw = np.frombuffer(buffer.array(), dtype=np.uint8)
                          # الإطار بصيغة RGBA — إزالة التحشية إذا row_stride > screen_w*4
                          if len(raw) >= screen_h * row_stride:
                              frame = raw[:screen_h * row_stride].reshape(screen_h, row_stride // 4, 4)
                              frame = frame[:, :screen_w, :]  # حذف البيكسلات الزائدة

                              # تحويل RGBA -> BGR لـ OpenCV
                              if CV2_AVAILABLE:
                                  frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                              # استدعاء callback المعالجة
                              if self._callback:
                                  self._callback(frame)

                      image.close()

              except Exception as frame_err:
                  pass   # تجاهل أخطاء القراءة المفردة

              # انتظار الوقت المتبقي للإطار
              elapsed = time.time() - t0
              wait    = frame_interval - elapsed
              if wait > 0:
                  time.sleep(wait)

      def _start_demo_mode(self):
          """وضع التجريب: توليد إطارات اصطناعية على سطح المكتب."""
          print("[ScreenCapture] Demo mode — synthetic frames")

          def _generate_demo():
              if not NUMPY_AVAILABLE:
                  return
              while True:
                  frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                  frame[:, :, 0] = 20   # خلفية داكنة مائلة للأزرق
                  frame[:, :, 2] = 40
                  if self._callback:
                      self._callback(frame)
                  time.sleep(1.0 / 30.0)

          t = threading.Thread(target=_generate_demo, name="ZekraDemoFrame", daemon=True)
          t.start()

      def stop(self):
          """إيقاف التقاط الشاشة وتحرير الموارد."""
          self._running = False
          if self._virtual_disp:
              try:
                  self._virtual_disp.release()
              except Exception:
                  pass
          if self._projection:
              try:
                  self._projection.stop()
              except Exception:
                  pass
          print("[ScreenCapture] Stopped and resources released")


  # ===== التطبيق الرئيسي =====

  class ZekraApp(App):
      """
      التطبيق الرئيسي لـ Zekra AI.
      يدير دورة حياة التطبيق كاملة:
      - طلب الصلاحيات
      - تهيئة محرك الرؤية
      - بدء التقاط الشاشة
      - رسم مربعات التتبع على الطبقة العلوية
      """

      def build(self):
          # إعداد النافذة — شفافة لطبقة Overlay
          Window.clearcolor = (0, 0, 0, 0)

          # على أندرويد، نجعل النافذة تظهر فوق كل شيء
          if IS_ANDROID:
              try:
                  from jnius import autoclass
                  LayoutParams = autoclass('android.view.WindowManager$LayoutParams')
                  mActivity.getWindow().setType(LayoutParams.TYPE_APPLICATION_OVERLAY)
              except Exception:
                  pass

          self._engine:  Optional[ProfessionalVisionEngine] = None
          self._capture: Optional[ScreenCaptureManager]     = None
          self._overlay: Optional[ZekraOverlay]             = None

          # بناء الواجهة الرئيسية
          self._root = ZekraMainLayout()
          self._perm_screen = PermissionScreen()
          self._root.add_widget(self._perm_screen)

          # التحقق من الصلاحيات فوراً
          Clock.schedule_once(self._check_permissions, 0.1)

          return self._root

      # ==================== إدارة الصلاحيات ====================

      def _check_permissions(self, dt):
          """فحص الصلاحيات عند البدء — إذا ممنوحة نبدأ مباشرة."""
          if not IS_ANDROID or not ANDROID_PERM_AVAILABLE:
              # سطح المكتب — نبدأ مباشرة
              Clock.schedule_once(lambda dt2: self.on_permissions_granted(), 0.5)
              return

          try:
              cam_ok = check_permission(Permission.CAMERA)
              stor_ok = (check_permission(Permission.READ_EXTERNAL_STORAGE) and
                         check_permission(Permission.WRITE_EXTERNAL_STORAGE))

              if cam_ok and stor_ok:
                  Clock.schedule_once(lambda dt2: self.on_permissions_granted(), 0.1)
              else:
                  self._perm_screen.perm_message = (
                      "⚠️ يحتاج التطبيق إلى صلاحيات الكاميرا والتخزين\n"
                      "اضغط الزر أدناه للمتابعة"
                  )
          except Exception as e:
              print(f"[Zekra] Permission check error: {e}")
              Clock.schedule_once(lambda dt2: self.on_permissions_granted(), 0.5)

      def on_permissions_granted(self):
          """نُستدعى بعد منح جميع الصلاحيات — نبني الواجهة الكاملة ونبدأ."""
          print("[Zekra] All permissions granted — initializing full UI")

          # إزالة شاشة الصلاحيات
          self._root.remove_widget(self._perm_screen)

          # بناء واجهة التشغيل
          self._build_runtime_ui()

          # تهيئة محرك الرؤية
          self._init_engine()

          # بدء التقاط الشاشة
          self._start_capture()

          # بدء حلقة رسم مربعات التتبع (30 مرة/ثانية)
          Clock.schedule_interval(self._render_loop, 1.0 / 30.0)

      # ==================== بناء واجهة التشغيل ====================

      def _build_runtime_ui(self):
          """بناء الواجهة الرئيسية بعد منح الصلاحيات."""
          # طبقة رسم مربعات التتبع
          self._overlay = ZekraOverlay(size=Window.size, pos=(0, 0))
          self._root.add_widget(self._overlay)

          # شريط الحالة العلوي
          self._status_bar = StatusBar(
              size_hint=(1, None),
              height=48,
              pos=(0, Window.height - 48)
          )
          self._status_bar.status_text = "🔵 Zekra AI — يعمل"
          self._root.add_widget(self._status_bar)

          # ربط تغيير حجم النافذة
          Window.bind(size=self._on_window_resize)

      def _on_window_resize(self, win, size):
          if self._overlay:
              self._overlay.size = size
          if self._status_bar:
              self._status_bar.pos = (0, size[1] - 48)

      # ==================== تهيئة المحرك ====================

      def _init_engine(self):
          """تهيئة محرك الرؤية الذكي."""
          if ENGINE_AVAILABLE:
              try:
                  self._engine = ProfessionalVisionEngine(
                      model_path='yolov8n.onnx',
                      input_size=(640, 640),
                      zoom_factor=1.5
                  )
                  print("[Zekra] Vision engine initialized")
              except Exception as e:
                  print(f"[Zekra] Engine init error: {e}")
                  self._engine = None
          else:
              print("[Zekra] Engine not available — demo mode")

      # ==================== التقاط الشاشة ====================

      def _start_capture(self):
          """بدء التقاط بث الشاشة الحي."""
          self._capture = ScreenCaptureManager(self._on_frame_received)
          self._capture.request_capture()

      def _on_frame_received(self, frame):
          """
          Callback يُستدعى لكل إطار من الشاشة.
          يُرسل الإطار لمحرك الرؤية ويحدّث قائمة الأجسام المتتبَّعة.
          """
          if self._engine is None:
              return

          try:
              tracked = self._engine.process_frame(frame)
              # التحديث يتم في الـ overlay عند حلقة الرسم
              self._last_tracked = tracked
          except Exception as e:
              print(f"[Zekra] Frame processing error: {e}")

      # ==================== حلقة الرسم ====================

      _last_tracked = []

      def _render_loop(self, dt):
          """
          تُستدعى 30 مرة/ثانية لتحديث مربعات التتبع على الشاشة.
          آمنة تماماً من الـ main thread.
          """
          if self._overlay is None:
              return

          # تحديث الطبقة العائمة
          self._overlay.update_tracked(self._last_tracked)
          self._overlay._redraw()

          # تحديث شريط الحالة
          if self._engine and self._status_bar:
              count = len(self._last_tracked)
              fps   = getattr(self._engine, 'fps', 0.0)
              self._status_bar.fps_text    = f"{fps:.1f} FPS"
              self._status_bar.status_text = (
                  f"🔵 Zekra AI — {count} هدف مرصود"
                  if count > 0 else
                  "🟡 Zekra AI — لا أهداف"
              )

      # ==================== دورة حياة التطبيق ====================

      def on_stop(self):
          """تنظيف الموارد عند إغلاق التطبيق."""
          if self._capture:
              self._capture.stop()
          print("[Zekra] Application stopped cleanly")


  # ===== نقطة الدخول =====
  if __name__ == '__main__':
      ZekraApp().run()
  