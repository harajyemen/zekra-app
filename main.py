"""
On-Device AI Camera Processor - Main Application with Live Stream & Overlay
===========================================================================
Professional Kivy mobile application for real-time object detection.
Highly Stable & Safe Bootstrapped Edition to prevent Android Crashes.

Author: Qusai Mohammed Jadelan & AI Team
Version: 1.2.0 (Super Safe Boot)
"""

import os
import sys
import time
from typing import Optional, List, Tuple

# تأمين مكتبة النمباي لمنع الانهيار أثناء التجميع
try:
    import numpy as np
except ImportError:
    np = None

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.utils import platform

# حماية استدعاء محرك الرؤية لمنع الانهيار عند التحميل
try:
    from offline_engine import ProfessionalVisionEngine, Detection
except Exception as e:
    ProfessionalVisionEngine = None
    print(f"[Zekra AI] Safe Boot Note: offline_engine deferred. ({e})")

# محاكاة مكتبة OpenCV بشكل آمن جداً لمنع كراش الاستيراد (Import Crash)
try:
    import cv2
except ImportError:
    class MockCV2:
        COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_BGR2RGBA, COLOR_RGB2BGR = 6, 4, 2, 4
        THRESH_BINARY, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, MORPH_ELLIPSE, MORPH_CLOSE, MORPH_OPEN, INTER_LINEAR = 0, 0, 1, 2, 3, 2, 1
        @staticmethod
        def cvtColor(img, code): return img
        @staticmethod
        def GaussianBlur(img, ksize, sigma): return img
        @staticmethod
        def absdiff(a, b): 
            if a is None or b is None: return np.zeros((10,10), dtype=np.uint8)
            return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
        @staticmethod
        def threshold(img, thresh, maxval, type_): return thresh, img
        @staticmethod
        def getStructuringElement(shape, ksize): return np.ones(ksize, dtype=np.uint8)
        @staticmethod
        def morphologyEx(img, op, kernel): return img
        @staticmethod
        def findContours(img, mode, method): return [], None
        @staticmethod
        def boundingRect(contour): return (0, 0, 10, 10)
        @staticmethod
        def resize(img, dsize, interpolation=1): return img[:dsize[1], :dsize[0]]
        @staticmethod
        def calcOpticalFlowFarneback(*args): return np.zeros((10, 10, 2), dtype=np.float32)
    cv2 = MockCV2()

# تصميم واجهة المستخدم بلغة الـ KV المدمجة والمؤمنة
KV = '''
#:kivy 2.2.0

<OverlayWidget>:
    canvas.after:
        Color: rgba=(0, 0, 0, 0.5)
        Rectangle:
            pos: self.pos
            size: self.width, dp(48)
        Color: rgba=(1, 1, 1, 0.3)
        Line:
            points: [self.x, self.y + dp(48), self.x + self.width, self.y + dp(48)]
            width: 1

<CameraPreview>:
    orientation: 'vertical'
    canvas.before:
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            pos: self.pos
            size: self.size

<InfoPanel>:
    canvas.before:
        Color:
            rgba: 0.1, 0.1, 0.1, 0.85
        Rectangle:
            pos: self.pos
            size: self.size
    orientation: 'vertical'
    padding: dp(10)
    spacing: dp(5)

<MainScreen>:
    orientation: 'vertical'

    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: 0.08
        padding: dp(5)
        spacing: dp(10)
        canvas.before:
            Color:
                rgba: 0.15, 0.15, 0.15, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Button:
            text: 'صلاحيات النافذة العائمة'
            font_size: sp(11)
            background_color: 0.2, 0.6, 1, 1
            on_press: app.open_overlay_settings()

        Button:
            id: stream_btn
            text: 'بدء بث / تسجيل الشاشة'
            font_size: sp(11)
            background_color: 1, 0.2, 0.2, 1
            on_press: app.toggle_screen_streaming()

    CameraPreview:
        id: camera_preview
        size_hint_y: 0.84

    InfoPanel:
        id: info_panel
        size_hint_y: 0.08
        spacing: dp(2)

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.5

            Label:
                id: fps_label
                text: 'FPS: --'
                font_size: sp(12)
                color: 0, 1, 0.5, 1
                size_hint_x: 0.25
                halign: 'left'
                text_size: self.size

            Label:
                id: detections_label
                text: 'Detections: 0'
                font_size: sp(12)
                color: 1, 1, 1, 1
                size_hint_x: 0.25
                halign: 'center'
                text_size: self.size

            Label:
                id: motion_label
                text: 'Motion: --'
                font_size: sp(12)
                color: 1, 0.8, 0, 1
                size_hint_x: 0.25
                halign: 'center'
                text_size: self.size

            Label:
                id: status_label
                text: 'Status: Loading Base UI...'
                font_size: sp(12)
                color: 1, 1, 0, 1
                size_hint_x: 0.25
                halign: 'right'
                text_size: self.size

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.5

            Label:
                id: inference_label
                text: 'Inference: --'
                font_size: sp(10)
                color: 0.7, 0.7, 1, 1
                size_hint_x: 0.33
                halign: 'left'
                text_size: self.size

            Label:
                id: resolution_label
                text: 'Resolution: --'
                font_size: sp(10)
                color: 0.7, 0.7, 1, 1
                size_hint_x: 0.33
                halign: 'center'
                text_size: self.size

            Label:
                id: model_label
                text: 'Model: YOLOv8n (Safe Mode)'
                font_size: sp(10)
                color: 0.7, 0.7, 1, 1
                size_hint_x: 0.34
                halign: 'right'
                text_size: self.size
'''


class OverlayWidget(Widget):
    detections: list = []
    preview_width: int = 0
    preview_height: int = 0
    frame_width: int = 640
    frame_height: int = 480

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self._update_canvas, size=self._update_canvas)

    def update_detections(self, detections: list) -> None:
        self.detections = detections
        self._update_canvas()

    def update_dimensions(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _scale_coordinates(self, x: int, y: int, source_width: int, source_height: int) -> Tuple[float, float]:
        widget_aspect = self.preview_width / max(self.preview_height, 1)
        frame_aspect = source_width / max(source_height, 1)

        if widget_aspect > frame_aspect:
            display_height = self.preview_height
            display_width = self.preview_height * frame_aspect
            offset_x = (self.preview_width - display_width) / 2
            offset_y = 0
        else:
            display_width = self.preview_width
            display_height = self.preview_width / frame_aspect
            offset_x = 0
            offset_y = (self.preview_height - display_height) / 2

        scaled_x = offset_x + (x / source_width) * display_width
        scaled_y = offset_y + (y / source_height) * display_height
        return scaled_x, scaled_y

    def _update_canvas(self, *args) -> None:
        self.canvas.after.clear()
        self.preview_width = int(self.width)
        self.preview_height = int(self.height)

        if self.preview_width == 0 or self.preview_height == 0 or not self.detections:
            return

        with self.canvas.after:
            for det in self.detections:
                try:
                    x1, y1, x2, y2 = det.bbox
                    center_x, center_y = det.center

                    scaled_x1, scaled_y1 = self._scale_coordinates(x1, y1, self.frame_width, self.frame_height)
                    scaled_x2, scaled_y2 = self._scale_coordinates(x2, y2, self.frame_width, self.frame_height)
                    scaled_cx, scaled_cy = self._scale_coordinates(center_x, center_y, self.frame_width, self.frame_height)

                    box_width = scaled_x2 - scaled_x1
                    box_height = scaled_y2 - scaled_y1

                    if hasattr(det, 'class_name') and det.class_name == 'person':
                        Color(1, 0.2, 0.2, 0.9)
                    elif hasattr(det, 'class_name') and det.class_name == 'car':
                        Color(0.2, 0.6, 1, 0.9)
                    else:
                        Color(1, 1, 0.2, 0.9)

                    Rectangle(pos=(scaled_x1, scaled_y1), size=(box_width, box_height))
                    Color(1, 1, 1, 1)
                    Line(rectangle=(scaled_x1, scaled_y1, box_width, box_height), width=2)
                except Exception:
                    pass


class CameraPreview(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera: Optional[Camera] = None
        self.overlay: Optional[OverlayWidget] = None
        Clock.schedule_once(self._init_camera, 1.0)

    def _init_camera(self, dt: float) -> None:
        try:
            # تهيئة الكاميرا بشكل محمي تماماً لتفادي كراش الهواتف بدون صلاحية فورية
            self.camera = Camera(index=0, resolution=(640, 480), play=True, size_hint=(1, 1))
            self.add_widget(self.camera)
            self.overlay = OverlayWidget(size_hint=(1, 1), pos_hint={'x': 0, 'y': 0})
            self.add_widget(self.overlay)
        except Exception as e:
            print(f"[Zekra Camera] Bypass layout hook: {e}")

    def get_frame_array(self) -> Optional[np.ndarray]:
        if np is None or not self.camera or not self.camera.texture:
            return None
        try:
            texture = self.camera.texture
            pixels = texture.pixels
            frame_rgba = np.frombuffer(pixels, dtype=np.uint8).reshape((texture.height, texture.width, 4))
            return frame_rgba[:, :, :3][:, :, ::-1].copy()
        except Exception:
            return None

    def get_texture_size(self) -> Tuple[int, int]:
        if self.camera and self.camera.texture:
            return (self.camera.texture.width, self.camera.texture.height)
        return (640, 480)

    def update_overlay(self, detections: list) -> None:
        if self.overlay:
            frame_width, frame_height = self.get_texture_size()
            self.overlay.update_dimensions(frame_width, frame_height)
            self.overlay.update_detections(detections)


class InfoPanel(BoxLayout):
    pass


class MainScreen(BoxLayout):
    pass


class AICameraProcessorApp(App):
    fps_value = NumericProperty(0)
    detections_count = NumericProperty(0)
    is_processing = BooleanProperty(False)
    status_text = StringProperty('Initializing...')
    is_streaming = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision_engine = None
        self.processing_clock = None
        self.processing_interval = 1.0 / 15.0

    def build(self):
        Window.fullscreen = 'auto'
        # تحميل واجهة المانيفست الرسومية أولاً لحماية التطبيق من الكراش
        self.root = Builder.load_string(KV)
        MainScreenView = MainScreen()
        
        # جدولة ذكية ومتباعدة لمنع الصدمة البرمجية عند الإقلاع
        Clock.schedule_once(self._initialize_engine, 2.5)
        Clock.schedule_once(self._ask_android_permissions, 4.0)
        return MainScreenView

    def _ask_android_permissions(self, dt: float):
        if platform == 'android':
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([
                    Permission.CAMERA,
                    Permission.RECORD_AUDIO
                ])
            except Exception as e:
                print(f"[Permissions] Deferred: {e}")

    def open_overlay_settings(self):
        if platform == 'android':
            try:
                from jnius import autoclass
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                Intent = autoclass('android.content.Intent')
                Settings = autoclass('android.provider.Settings')
                Uri = autoclass('android.net.Uri')
                
                activity = PythonActivity.mActivity
                if not Settings.canDrawOverlays(activity):
                    intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:" + activity.getPackageName()))
                    activity.startActivity(intent)
                    self._update_status("قم بتفعيل الصلاحية ثم عد للتطبيق")
                else:
                    self._update_status("صلاحية النافذة العائمة مفعلة مسبقاً ✅")
            except Exception as e:
                self._update_status(f"خطأ في فتح الإعدادات: {str(e)[:20]}")

    def toggle_screen_streaming(self):
        if platform != 'android':
            self._update_status("البث متاح فقط على الأندرويد")
            return

        try:
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            stream_btn = self.root.ids.get('stream_btn')

            try:
                service = autoclass('org.offline.aicameraprocessor.ServiceMyservice')
            except Exception:
                self._update_status("خطأ: لم يتم العثور على حزمة الخدمة")
                return

            if not self.is_streaming:
                self.is_streaming = True
                if stream_btn:
                    stream_btn.text = "إيقاف البث المباشر 🛑"
                    stream_btn.background_color = (1, 0.5, 0, 1)
                self._update_status("جاري تشغيل خدمة البث...")
                service.start(activity, "")
            else:
                self.is_streaming = False
                if stream_btn:
                    stream_btn.text = "بدء بث / تسجيل الشاشة 📡"
                    stream_btn.background_color = (1, 0.2, 0.2, 1)
                service.stop(activity)
                self._update_status("تم إيقاف خدمة البث")
        except Exception as e:
            self._update_status(f"فشلت الخدمة: {str(e)[:15]}")

    def _initialize_engine(self, dt: float) -> None:
        # إذا لم يتم استيراد المحرك، يتحول الوضع آلياً إلى نمط الأمان والمراقب بدون كراش
        if ProfessionalVisionEngine is None:
            self._update_status('Safe Mode (AI Bypassed)')
            return
            
        try:
            model_path = 'yolov8n.onnx'
            if not os.path.exists(model_path):
                self._update_status('Notice: ONNX Model missing')
                return

            self.vision_engine = ProfessionalVisionEngine(
                model_path=model_path, input_size=(640, 640), use_gpu=False
            )
            
            if self.vision_engine.is_initialized:
                self._update_status('AI Engine Ready')
                self.processing_clock = Clock.schedule_interval(self._process_frame, self.processing_interval)
            else:
                self._update_status('AI Engine Safe Mode')
        except Exception as e:
            self._update_status(f'AI Hold: {str(e)[:15]}')

    def _process_frame(self, dt: float) -> None:
        if not self.vision_engine or not self.vision_engine.is_initialized:
            return
        camera_preview = self.root.ids.get('camera_preview')
        if not camera_preview:
            return
        frame = camera_preview.get_frame_array()
        if frame is None or frame.size == 0:
            return

        try:
            detections, metadata = self.vision_engine.process_frame(frame)
            camera_preview.update_overlay(detections)
            self._update_labels(
                fps=f"FPS: {metadata.get('fps', 0):.1f}",
                detections=f"Detections: {len(detections)}",
                motion=f"Motion: {metadata.get('motion_magnitude', 0):.1f}px"
            )
        except Exception:
            pass

    def _update_status(self, text: str) -> None:
        try:
            status_label = self.root.ids.get('status_label')
            if status_label:
                status_label.text = f'Status: {text}'
        except Exception:
            pass

    def _update_labels(self, fps=None, detections=None, motion=None) -> None:
        try:
            if fps: self.root.ids.get('fps_label').text = fps
            if detections: self.root.ids.get('detections_label').text = detections
            if motion: self.root.ids.get('motion_label').text = motion
        except Exception:
            pass

    def on_pause(self):
        if self.processing_clock: self.processing_clock.cancel()
        return True

    def on_resume(self):
        if self.vision_engine and self.vision_engine.is_initialized:
            self.processing_clock = Clock.schedule_interval(self._process_frame, self.processing_interval)

    def on_stop(self):
        if self.processing_clock: self.processing_clock.cancel()


if __name__ == '__main__':
    try:
        AICameraProcessorApp().run()
    except Exception as fatal_error:
        print(f"[Fatal Crash Prevented] Root runtime bypassed: {fatal_error}")
