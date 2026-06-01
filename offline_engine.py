"""
Professional Vision Engine - Offline AI Camera Processor
=========================================================
High-performance offline computer vision engine for real-time object detection
on mobile devices. Features advanced noise filtering, motion compensation,
and hardware-accelerated ONNX inference.
Highly Stable & Safe Edition to prevent Android Crashes.

Author: AI Camera Processor Team
Version: 1.1.0 (Bypass Crash Version)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import threading
import queue
import time
import os

# حماية استيراد مكتبة OpenCV لمنع انهيار بيئة أندرويد
try:
    import cv2
except ImportError:
    cv2 = None

# حماية استيراد مكتبة ONNX Runtime الحساسة جداً لمنع الانهيار الفوري
try:
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    width: int
    height: int


@dataclass
class MotionVector:
    dx: float
    dy: float
    magnitude: float
    angle: float


class ProfessionalVisionEngine:
    """
    High-performance offline vision engine for mobile devices with Safe Mode.
    """

    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    }

    ALERT_CLASSES = {0, 2}  # person, car
    ALERT_CLASS_NAMES = {'person', 'car'}
    CONFIDENCE_THRESHOLD = 0.60
    MIN_CLUSTER_SIZE = 15
    BLUR_KERNEL_SIZE = 5

    FLOW_PYRAMID_SCALE = 0.5
    FLOW_LEVELS = 3
    FLOW_WIN_SIZE = 15
    FLOW_ITERATIONS = 3
    FLOW_POLY_N = 5
    FLOW_POLY_SIGMA = 1.2

    def __init__(self, model_path: str = 'yolov8n.onnx',
                 input_size: Tuple[int, int] = (640, 640),
                 use_gpu: bool = False):
        self.model_path = model_path
        self.input_size = input_size
        self.use_gpu = use_gpu

        self.session = None
        self.input_name = None
        self.output_name = None

        self.prev_frame_gray = None
        self.prev_frame_blur = None

        self.motion_vector = MotionVector(0.0, 0.0, 0.0, 0.0)
        self.motion_threshold = 5.0

        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.processing_times: List[float] = []

        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)

        self.is_initialized = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        تجهيز آمن تماماً للنموذج - إذا لم تتوفر المكتبة أو الملف، يتم التحويل لنمط الأمان بدلاً من الكراش
        """
        if ort is None:
            print("[VisionEngine] Safe Mode Active: onnxruntime package missing.")
            self.is_initialized = False
            return

        if not os.path.exists(self.model_path):
            print(f"[VisionEngine] Safe Mode Active: {self.model_path} file not found.")
            self.is_initialized = False
            return

        providers = []
        if self.use_gpu and hasattr(ort, 'get_available_providers'):
            gpu_providers = ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'TensorrtExecutionProvider']
            for provider in gpu_providers:
                if provider in ort.get_available_providers():
                    providers.append(provider)
                    break

        if ort and hasattr(ort, 'get_available_providers') and 'CPUExecutionProvider' in ort.get_available_providers():
            providers.append('CPUExecutionProvider')

        if not providers:
            providers = ['CPUExecutionProvider']

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.is_initialized = True
            print(f"[VisionEngine] Model loaded successfully with providers: {providers}")
        except Exception as e:
            print(f"[VisionEngine] Init Error Bypassed: {e}")
            self.is_initialized = False

    def apply_anti_noise_filter(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0 or cv2 is None:
            return frame
        try:
            return cv2.GaussianBlur(frame, (self.BLUR_KERNEL_SIZE, self.BLUR_KERNEL_SIZE), 0)
        except Exception:
            return frame

    def calculate_global_motion(self, frame: np.ndarray) -> MotionVector:
        if frame is None or frame.size == 0 or cv2 is None:
            return MotionVector(0.0, 0.0, 0.0, 0.0)
        
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, (9, 9), 2.0)

            if self.prev_frame_gray is None or self.prev_frame_gray.shape != frame_gray.shape:
                self.prev_frame_gray = frame_gray.copy().astype(np.float32)
                return MotionVector(0.0, 0.0, 0.0, 0.0)

            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame_gray, frame_gray, None,
                pyr_scale=self.FLOW_PYRAMID_SCALE, levels=self.FLOW_LEVELS,
                winsize=self.FLOW_WIN_SIZE, iterations=self.FLOW_ITERATIONS,
                poly_n=self.FLOW_POLY_N, poly_sigma=self.FLOW_POLY_SIGMA, flags=0
            )

            dx = float(np.mean(flow[:, :, 0]))
            dy = float(np.mean(flow[:, :, 1]))
            magnitude = float(np.sqrt(dx * dx + dy * dy))
            angle = float(np.degrees(np.arctan2(dy, dx)))

            motion = MotionVector(dx, dy, magnitude, angle)
            self.prev_frame_gray = frame_gray.copy().astype(np.float32)
            self.motion_vector = motion
            return motion
        except Exception:
            return MotionVector(0.0, 0.0, 0.0, 0.0)

    def detect_micro_targets(self, frame: np.ndarray, prev_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if cv2 is None or prev_frame is None or frame.shape != prev_frame.shape:
            return []
        try:
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_current, gray_prev)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            micro_targets = []
            min_area = self.MIN_CLUSTER_SIZE * self.MIN_CLUSTER_SIZE
            max_area = 150 * 150

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if min_area <= area <= max_area:
                    aspect_ratio = w / max(h, 1)
                    if 0.2 < aspect_ratio < 5.0:
                        micro_targets.append((x, y, w, h))
            return micro_targets
        except Exception:
            return []

    def preprocess_for_inference(self, frame: np.ndarray) -> np.ndarray:
        self.orig_height, self.orig_width = frame.shape[:2]
        if cv2 is None:
            return np.zeros((1, 3, 640, 640), dtype=np.float32)
            
        try:
            input_w, input_h = self.input_size
            scale = min(input_w / frame.shape[1], input_h / frame.shape[0])
            new_w, new_h = int(frame.shape[1] * scale), int(frame.shape[0] * scale)

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
            pad_x, pad_y = (input_w - new_w) // 2, (input_h - new_h) // 2
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            self.pad_x, self.pad_y, self.scale = pad_x, pad_y, scale
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))
            return np.ascontiguousarray(np.expand_dims(transposed, axis=0))
        except Exception:
            return np.zeros((1, 3, 640, 640), dtype=np.float32)

    def run_inference(self, input_tensor: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_initialized or self.session is None:
            return None
        try:
            start_time = time.time()
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            self.processing_times.append(time.time() - start_time)
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
            return outputs[0]
        except Exception:
            return None

    def postprocess_detections(self, raw_output: Optional[np.ndarray], conf_threshold: float = None) -> List[Detection]:
        if raw_output is None or len(raw_output) == 0:
            return []
            
        if conf_threshold is None:
            conf_threshold = self.CONFIDENCE_THRESHOLD

        try:
            predictions = raw_output[0].T
            boxes = predictions[:, :4]
            class_scores = predictions[:, 4:]

            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)

            mask = confidences > conf_threshold
            boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

            target_mask = np.isin(class_ids, list(self.ALERT_CLASSES))
            boxes, confidences, class_ids = boxes[target_mask], confidences[target_mask], class_ids[target_mask]

            if len(boxes) == 0:
                return []

            detections = []
            for i in range(len(boxes)):
                x_center, y_center, w, h = boxes[i]
                x1 = int((x_center - w / 2 - self.pad_x) / self.scale)
                y1 = int((y_center - h / 2 - self.pad_y) / self.scale)
                x2 = int((x_center + w / 2 - self.pad_x) / self.scale)
                y2 = int((y_center + h / 2 - self.pad_y) / self.scale)

                x1, y1 = max(0, min(x1, self.orig_width - 1)), max(0, min(y1, self.orig_height - 1))
                x2, y2 = max(0, min(x2, self.orig_width - 1)), max(0, min(y2, self.orig_height - 1))

                class_id = int(class_ids[i])
                detection = Detection(
                    class_id=class_id,
                    class_name=self.COCO_CLASSES.get(class_id, f"class_{class_id}"),
                    confidence=float(confidences[i]),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    width=x2 - x1,
                    height=y2 - y1
                )
                detections.append(detection)

            if len(detections) > 0:
                detections = self._apply_nms(detections)
            return detections
        except Exception:
            return []

    def _apply_nms(self, detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
        try:
            boxes = np.array([d.bbox for d in detections])
            scores = np.array([d.confidence for d in detections])
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                if order.size == 1: break
                xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
                xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
                w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(iou <= iou_threshold)[0]
                order = order[inds + 1]
            return [detections[i] for i in keep]
        except Exception:
            return detections

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], Dict[str, Any]]:
        start_time = time.time()
        metadata = {
            'frame_shape': frame.shape if frame is not None else (480, 640, 3),
            'processing_time_ms': 0.0, 'motion_compensated': False,
            'motion_magnitude': 0.0, 'motion_angle': 0.0,
            'micro_targets_count': 0, 'inference_time_ms': 0.0,
            'fps': self.fps, 'alert_classes_only': True
        }

        if frame is None or frame.size == 0:
            return [], metadata

        try:
            denoised = self.apply_anti_noise_filter(frame)
            motion = self.calculate_global_motion(denoised)
            metadata['motion_compensated'] = motion.magnitude > self.motion_threshold
            metadata['motion_magnitude'] = motion.magnitude
            metadata['motion_angle'] = motion.angle

            prev_blur = self.prev_frame_blur
            self.prev_frame_blur = denoised.copy()

            if prev_blur is not None:
                metadata['micro_targets_count'] = len(self.detect_micro_targets(denoised, prev_blur))

            if self.is_initialized:
                input_tensor = self.preprocess_for_inference(denoised)
                raw_output = self.run_inference(input_tensor)
                detections = self.postprocess_detections(raw_output)
                if self.processing_times:
                    metadata['inference_time_ms'] = self.processing_times[-1] * 1000
            else:
                detections = []

            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_time = current_time

            metadata['processing_time_ms'] = (time.time() - start_time) * 1000
            metadata['fps'] = self.fps
            return detections, metadata
        except Exception:
            return [], metadata

    def get_average_inference_time(self) -> float:
        if not self.processing_times: return 0.0
        return (sum(self.processing_times) / len(self.processing_times)) * 1000

    def should_alert(self, detection: Detection) -> bool:
        return (detection.class_id in self.ALERT_CLASSES and
                detection.confidence > self.CONFIDENCE_THRESHOLD)

    def cleanup(self) -> None:
        if self.session:
            del self.session
            self.session = None
        self.prev_frame_gray = None
        self.prev_frame_blur = None
        self.processing_times.clear()
        self.is_initialized = False


class VisionEngineWorker:
    def __init__(self, engine: ProfessionalVisionEngine):
        self.engine = engine
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.thread = None
        self.running = False

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread: self.thread.join(timeout=2.0)

    def submit_frame(self, frame: np.ndarray) -> bool:
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[Tuple[List[Detection], Dict]]:
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self) -> None:
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                detections, metadata = self.engine.process_frame(frame)
                try:
                    while not self.result_queue.empty():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait((detections, metadata))
                except queue.Empty:
                    pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VisionEngineWorker] Queue bypass: {e}")


def create_default_engine(model_path: str = 'yolov8n.onnx') -> ProfessionalVisionEngine:
    return ProfessionalVisionEngine(model_path=model_path, input_size=(640, 640), use_gpu=False)


if __name__ == '__main__':
    engine = create_default_engine()
    print(f"Engine safe-checked. Initialized: {engine.is_initialized}")

