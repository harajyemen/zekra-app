"""
On-Device AI Camera Processor - Main Application
=================================================
Professional Kivy mobile application for real-time object detection
with offline AI processing. Features fullscreen camera preview with
real-time bounding box overlays and tracking crosshairs.

Author: AI Camera Processor Team
Version: 1.0.0
"""

import os
import sys
import time
from typing import Optional, List, Tuple

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.lang import Builder

# Import the vision engine
from offline_engine import ProfessionalVisionEngine, Detection, create_default_engine


# Kivy KV language definition for the UI
KV = '''
#:kivy 2.2.0

<OverlayWidget>:
    canvas.after:
        # Semi-transparent background for status bar
        Color: rgba=(0, 0, 0, 0.5)
        Rectangle:
            pos: self.pos
            size: self.width, dp(48)
        # Border line
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

    CameraPreview:
        id: camera_preview
        size_hint_y: 0.92

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
                text: 'Status: Initializing...'
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
                text: 'Inference: -- ms'
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
                text: 'Model: YOLOv8n'
                font_size: sp(10)
                color: 0.7, 0.7, 1, 1
                size_hint_x: 0.34
                halign: 'right'
                text_size: self.size
'''

Builder.load_string(KV)


class OverlayWidget(Widget):
    """
    Widget for drawing bounding boxes and tracking crosshairs
    directly over the camera preview.
    """

    # Active detections list
    detections: List[Detection] = []

    # Camera preview dimensions (updated from parent)
    preview_width: int = 0
    preview_height: int = 0

    # Frame dimensions (from camera)
    frame_width: int = 640
    frame_height: int = 480

    def __init__(self, **kwargs):
        """Initialize overlay widget."""
        super().__init__(**kwargs)
        self.bind(pos=self._update_canvas, size=self._update_canvas)

    def update_detections(self, detections: List[Detection]) -> None:
        """
        Update the detections list and redraw overlays.

        Args:
            detections: List of Detection objects to display
        """
        self.detections = detections
        self._update_canvas()

    def update_dimensions(self, frame_width: int, frame_height: int) -> None:
        """
        Update frame dimensions for coordinate scaling.

        Args:
            frame_width: Width of camera frames
            frame_height: Height of camera frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _scale_coordinates(self, x: int, y: int,
                           source_width: int, source_height: int) -> Tuple[float, float]:
        """
        Scale coordinates from source frame to widget coordinates.

        Args:
            x: X coordinate in source frame
            y: Y coordinate in source frame
            source_width: Width of source frame
            source_height: Height of source frame

        Returns:
            Tuple of (scaled_x, scaled_y) in widget coordinates
        """
        # Calculate aspect ratios
        widget_aspect = self.preview_width / max(self.preview_height, 1)
        frame_aspect = source_width / max(source_height, 1)

        # Calculate display dimensions (letterboxing)
        if widget_aspect > frame_aspect:
            # Widget is wider - bars on sides
            display_height = self.preview_height
            display_width = self.preview_height * frame_aspect
            offset_x = (self.preview_width - display_width) / 2
            offset_y = 0
        else:
            # Widget is taller - bars on top/bottom
            display_width = self.preview_width
            display_height = self.preview_width / frame_aspect
            offset_x = 0
            offset_y = (self.preview_height - display_height) / 2

        # Scale coordinates
        scaled_x = offset_x + (x / source_width) * display_width
        scaled_y = offset_y + (y / source_height) * display_height

        return scaled_x, scaled_y

    def _update_canvas(self, *args) -> None:
        """Redraw all overlays on canvas."""
        self.canvas.after.clear()

        # Update preview dimensions
        self.preview_width = int(self.width)
        self.preview_height = int(self.height)

        if self.preview_width == 0 or self.preview_height == 0:
            return

        with self.canvas.after:
            # Draw bounding boxes for each detection
            for det in self.detections:
                x1, y1, x2, y2 = det.bbox
                center_x, center_y = det.center

                # Scale coordinates to widget
                scaled_x1, scaled_y1 = self._scale_coordinates(
                    x1, y1, self.frame_width, self.frame_height
                )
                scaled_x2, scaled_y2 = self._scale_coordinates(
                    x2, y2, self.frame_width, self.frame_height
                )
                scaled_cx, scaled_cy = self._scale_coordinates(
                    center_x, center_y, self.frame_width, self.frame_height
                )

                # Calculate box dimensions
                box_width = scaled_x2 - scaled_x1
                box_height = scaled_y2 - scaled_y1

                # Color based on class (red for alerts)
                if det.class_name == 'person':
                    Color(1, 0.2, 0.2, 0.9)  # Red
                elif det.class_name == 'car':
                    Color(0.2, 0.6, 1, 0.9)  # Blue
                else:
                    Color(1, 1, 0.2, 0.9)  # Yellow

                # Draw solid bounding box (filled rectangle)
                Rectangle(
                    pos=(scaled_x1, scaled_y1),
                    size=(box_width, box_height)
                )

                # Draw bounding box outline (white, slightly larger)
                Color(1, 1, 1, 1)
                Line(
                    rectangle=(scaled_x1, scaled_y1, box_width, box_height),
                    width=2
                )

                # Draw tracking crosshair at center
                Color(0, 1, 0, 1)  # Green
                crosshair_size = min(box_width, box_height) * 0.15

                # Horizontal crosshair line
                Line(
                    points=[
                        scaled_cx - crosshair_size, scaled_cy,
                        scaled_cx + crosshair_size, scaled_cy
                    ],
                    width=2
                )

                # Vertical crosshair line
                Line(
                    points=[
                        scaled_cx, scaled_cy - crosshair_size,
                        scaled_cx, scaled_cy + crosshair_size
                    ],
                    width=2
                )

                # Draw center dot
                Ellipse(
                    pos=(scaled_cx - 4, scaled_cy - 4),
                    size=(8, 8)
                )

                # Draw confidence label
                Color(1, 1, 1, 1)
                label_text = f"{det.class_name}: {det.confidence:.0%}"

                # Label background
                label_bg_width = len(label_text) * 9
                label_bg_height = 18
                Color(0, 0, 0, 0.7)
                Rectangle(
                    pos=(scaled_x1, scaled_y2 + 2),
                    size=(label_bg_width, label_bg_height)
                )

                # Label position indicator
                Color(1, 1, 0, 1)
                Line(
                    points=[scaled_x1, scaled_y2, scaled_x1 + 15, scaled_y2 + 15],
                    width=1
                )


class CameraPreview(BoxLayout):
    """
    Camera preview widget with integrated overlay for detections.
    """

    def __init__(self, **kwargs):
        """Initialize camera preview."""
        super().__init__(**kwargs)
        self.camera: Optional[Camera] = None
        self.overlay: Optional[OverlayWidget] = None
        self.camera_texture: Optional[Texture] = None

        # Frame processing state
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0  # Target 30 FPS

        # Schedule camera initialization
        Clock.schedule_once(self._init_camera, 0.5)

    def _init_camera(self, dt: float) -> None:
        """Initialize the camera widget."""
        try:
            # Create camera widget
            # Note: Kivy's Camera uses index 0 for the default camera
            self.camera = Camera(
                index=0,
                resolution=(640, 480),
                play=True,
                size_hint=(1, 1)
            )
            self.add_widget(self.camera)

            # Create overlay widget on top
            self.overlay = OverlayWidget(
                size_hint=(1, 1),
                pos_hint={'x': 0, 'y': 0}
            )
            self.add_widget(self.overlay)

            print("[CameraPreview] Camera initialized successfully")

        except Exception as e:
            print(f"[CameraPreview] Error initializing camera: {e}")

    def get_frame_array(self) -> Optional[np.ndarray]:
        """
        Extract the current camera frame as a NumPy array.

        Returns:
            BGR numpy array or None if texture unavailable
        """
        if not self.camera or not self.camera.texture:
            return None

        try:
            texture = self.camera.texture
            if texture is None:
                return None

            # Get texture buffer
            pixels = texture.pixels

            # Convert to numpy array
            # Kivy textures are RGBA, need to convert to BGR
            frame_rgba = np.frombuffer(pixels, dtype=np.uint8)
            frame_rgba = frame_rgba.reshape((texture.height, texture.width, 4))

            # Convert RGBA to BGR
            frame_bgr = cv2_color_convert_rgba_to_bgr(frame_rgba)

            return frame_bgr

        except Exception as e:
            print(f"[CameraPreview] Error extracting frame: {e}")
            return None

    def get_texture_size(self) -> Tuple[int, int]:
        """Get camera texture size (width, height)."""
        if self.camera and self.camera.texture:
            return (self.camera.texture.width, self.camera.texture.height)
        return (640, 480)

    def update_overlay(self, detections: List[Detection]) -> None:
        """Update detection overlay with new detections."""
        if self.overlay:
            frame_width, frame_height = self.get_texture_size()
            self.overlay.update_dimensions(frame_width, frame_height)
            self.overlay.update_detections(detections)


def cv2_color_convert_rgba_to_bgr(frame_rgba: np.ndarray) -> np.ndarray:
    """
    Convert RGBA frame to BGR without using cv2 (for Kivy compatibility).

    Args:
        frame_rgba: RGBA numpy array (H, W, 4)

    Returns:
        BGR numpy array (H, W, 3)
    """
    # Extract RGB channels (drop alpha)
    frame_rgb = frame_rgba[:, :, :3]

    # Convert RGB to BGR by reversing channel order
    frame_bgr = frame_rgb[:, :, ::-1].copy()

    return frame_bgr


class InfoPanel(BoxLayout):
    """Information panel displaying stats and status."""

    pass


class MainScreen(BoxLayout):
    """Main application screen containing camera and info panel."""

    def __init__(self, **kwargs):
        """Initialize main screen."""
        super().__init__(**kwargs)


class AICameraProcessorApp(App):
    """
    Main application class for the AI Camera Processor.

    Manages the vision engine and camera preview, coordinating
    real-time object detection and display updates.
    """

    # Properties for UI binding
    fps_value = NumericProperty(0)
    detections_count = NumericProperty(0)
    is_processing = BooleanProperty(False)
    status_text = StringProperty('Initializing...')
    motion_text = StringProperty('Motion: --')
    inference_text = StringProperty('Inference: -- ms')
    resolution_text = StringProperty('Resolution: --')

    def __init__(self, **kwargs):
        """Initialize the application."""
        super().__init__(**kwargs)

        # Vision engine
        self.vision_engine: Optional[ProfessionalVisionEngine] = None

        # Processing clock
        self.processing_clock = None

        # Frame counter for throttling
        self.frame_counter = 0

        # Processing interval (adjust for performance)
        self.processing_interval = 1.0 / 15.0  # Target 15 FPS inference

    def build(self):
        """Build the application UI."""
        # Set fullscreen mode
        Window.fullscreen = 'auto'

        # Create main screen from KV
        self.root = MainScreen()

        # Start initialization
        Clock.schedule_once(self._initialize_engine, 1.0)

        return self.root

    def _initialize_engine(self, dt: float) -> None:
        """Initialize the vision engine."""
        try:
            # Check for model file
            model_path = 'yolov8n.onnx'
            if not os.path.exists(model_path):
                self._update_status('Error: Model file not found')
                self._update_labels(
                    fps='--',
                    detections='Model: Missing',
                    motion='--',
                    inference='--',
                    resolution='--'
                )
                return

            # Create vision engine
            self.vision_engine = ProfessionalVisionEngine(
                model_path=model_path,
                input_size=(640, 640),
                use_gpu=False  # CPU for reliability on mobile
            )

            print(f"[App] Vision engine initialized: {self.vision_engine.is_initialized}")

            # Update status
            self._update_status('Engine Ready')
            self._update_labels(
                fps='FPS: 0',
                detections='Detections: 0',
                motion='Motion: 0',
                inference='Inference: 0 ms',
                resolution='Resolution: 640x480'
            )

            # Start processing loop
            self.processing_clock = Clock.schedule_interval(
                self._process_frame,
                self.processing_interval
            )

        except Exception as e:
            print(f"[App] Error initializing engine: {e}")
            self._update_status(f'Error: {str(e)[:30]}')

    def _process_frame(self, dt: float) -> None:
        """
        Process a camera frame through the vision engine.

        Called by the Clock scheduler at regular intervals.
        """
        if not self.vision_engine or not self.vision_engine.is_initialized:
            return

        # Get the camera preview widget
        camera_preview = self.root.ids.get('camera_preview')
        if not camera_preview:
            return

        # Get frame from camera
        frame = camera_preview.get_frame_array()
        if frame is None or frame.size == 0:
            return

        try:
            # Process frame through vision engine
            detections, metadata = self.vision_engine.process_frame(frame)

            # Update overlay with detections
            camera_preview.update_overlay(detections)

            # Update UI labels
            self._update_labels(
                fps=f"FPS: {metadata['fps']:.1f}",
                detections=f"Detections: {len(detections)}",
                motion=f"Motion: {metadata['motion_magnitude']:.1f}px",
                inference=f"Inference: {metadata['inference_time_ms']:.0f}ms",
                resolution=f"{metadata['frame_shape'][1]}x{metadata['frame_shape'][0]}"
            )

            # Update alert status
            alert_count = sum(1 for d in detections if self.vision_engine.should_alert(d))
            if alert_count > 0:
                self._update_status(f'ALERT: {alert_count} target(s)!')
            else:
                self._update_status('Monitoring...')

        except Exception as e:
            print(f"[App] Processing error: {e}")

    def _update_status(self, text: str) -> None:
        """Update status label."""
        status_label = self.root.ids.get('status_label')
        if status_label:
            status_label.text = f'Status: {text}'

            # Change color based on status
            if 'ALERT' in text:
                status_label.color = (1, 0.2, 0.2, 1)
            elif 'Error' in text:
                status_label.color = (1, 0, 0, 1)
            elif 'Ready' in text or 'Monitoring' in text:
                status_label.color = (0, 1, 0.5, 1)
            else:
                status_label.color = (1, 1, 0, 1)

    def _update_labels(self, fps: str = None, detections: str = None,
                       motion: str = None, inference: str = None,
                       resolution: str = None) -> None:
        """Update info panel labels."""
        if fps:
            fps_label = self.root.ids.get('fps_label')
            if fps_label:
                fps_label.text = fps

        if detections:
            det_label = self.root.ids.get('detections_label')
            if det_label:
                det_label.text = detections

        if motion:
            motion_label = self.root.ids.get('motion_label')
            if motion_label:
                motion_label.text = motion

        if inference:
            inf_label = self.root.ids.get('inference_label')
            if inf_label:
                inf_label.text = inference

        if resolution:
            res_label = self.root.ids.get('resolution_label')
            if res_label:
                res_label.text = resolution

    def on_pause(self):
        """Handle application pause."""
        if self.processing_clock:
            self.processing_clock.cancel()
        return True

    def on_resume(self):
        """Handle application resume."""
        if self.vision_engine and self.vision_engine.is_initialized:
            self.processing_clock = Clock.schedule_interval(
                self._process_frame,
                self.processing_interval
            )

    def on_stop(self):
        """Handle application stop."""
        if self.processing_clock:
            self.processing_clock.cancel()

        if self.vision_engine:
            self.vision_engine.cleanup()


# Platform-specific camera handling for Android
try:
    from jnius import autoclass

    # Android camera constants
    Camera = autoclass('android.hardware.Camera')
    CameraParameters = autoclass('android.hardware.Camera$Parameters')

    def get_camera_resolutions():
        """Get available camera resolutions on Android."""
        try:
            camera = Camera.open(0)
            parameters = camera.getParameters()
            sizes = parameters.getSupportedPreviewSizes()

            resolutions = []
            for size in sizes:
                resolutions.append((size.width, size.height))

            camera.release()
            return resolutions
        except Exception as e:
            print(f"[Android] Error getting resolutions: {e}")
            return [(640, 480)]

except ImportError:
    # Not on Android or jnius not available
    def get_camera_resolutions():
        """Default resolutions for non-Android platforms."""
        return [(640, 480), (1280, 720), (1920, 1080)]


# Mock cv2 for testing without full import
# In production, cv2 should be properly installed
try:
    import cv2
except ImportError:
    # Create mock for testing
    class MockCV2:
        COLOR_BGR2GRAY = 6
        COLOR_BGR2RGB = 4
        COLOR_BGR2RGBA = 2
        COLOR_RGB2BGR = 4
        THRESH_BINARY = 0
        RETR_EXTERNAL = 0
        CHAIN_APPROX_SIMPLE = 1
        MORPH_ELLIPSE = 2
        MORPH_CLOSE = 3
        MORPH_OPEN = 2
        INTER_LINEAR = 1

        @staticmethod
        def cvtColor(img, code):
            if len(img.shape) == 3:
                if code == 6:  # BGR2GRAY
                    return np.dot(img[...,:3], [0.114, 0.587, 0.299]).astype(np.uint8)
                elif code == 4:  # BGR2RGB
                    return img[:,:,::-1].copy()
            return img

        @staticmethod
        def GaussianBlur(img, ksize, sigma):
            return img

        @staticmethod
        def absdiff(a, b):
            return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)

        @staticmethod
        def threshold(img, thresh, maxval, type_):
            return thresh, np.where(img > thresh, maxval, 0).astype(np.uint8)

        @staticmethod
        def getStructuringElement(shape, ksize):
            return np.ones(ksize, dtype=np.uint8)

        @staticmethod
        def morphologyEx(img, op, kernel):
            return img

        @staticmethod
        def findContours(img, mode, method):
            return [], None

        @staticmethod
        def boundingRect(contour):
            return (0, 0, 10, 10)

        @staticmethod
        def resize(img, dsize, interpolation=INTER_LINEAR):
            return img[:dsize[1], :dsize[0]]

        @staticmethod
        def calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels,
                                     winsize, iterations, poly_n, poly_sigma, flags):
            return np.zeros((prev.shape[0], prev.shape[1], 2), dtype=np.float32)

    cv2 = MockCV2()


if __name__ == '__main__':
    # Run the application
    app = AICameraProcessorApp()
    app.run()
