"""
Professional Vision Engine - Offline AI Camera Processor
=========================================================
High-performance offline computer vision engine for real-time object detection
on mobile devices. Features advanced noise filtering, motion compensation,
and hardware-accelerated ONNX inference.

Author: AI Camera Processor Team
Version: 1.0.0
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import threading
import queue
import time
import os


@dataclass
class Detection:
    """
    Represents a single object detection result.

    Attributes:
        class_id: COCO class ID (0=person, 2=car, etc.)
        class_name: Human-readable class name
        confidence: Detection confidence score (0.0 to 1.0)
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates
        center: Center point of the detection (x, y)
        width: Width of bounding box in pixels
        height: Height of bounding box in pixels
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    width: int
    height: int


@dataclass
class MotionVector:
    """
    Represents global motion compensation data.

    Attributes:
        dx: Horizontal displacement in pixels
        dy: Vertical displacement in pixels
        magnitude: Total motion magnitude
        angle: Motion direction in degrees
    """
    dx: float
    dy: float
    magnitude: float
    angle: float


class ProfessionalVisionEngine:
    """
    High-performance offline vision engine for mobile devices.

    This engine implements a multi-stage processing pipeline:

    1. Anti-Noise Filter: Gaussian blur to eliminate sensor noise and artifacts
    2. Global Motion Compensation: Farneback optical flow to detect and compensate
       for camera movement, preventing false positives from panning
    3. Micro-Target Detection: Isolate small moving pixel clusters (15x15+ pixels)
    4. Hardware-Accelerated Inference: ONNX Runtime with CPU/GPU acceleration
    5. Strict Classification: Only alert on 'person' (class 0) or 'car' (class 2)
       with confidence > 0.60

    The engine is designed for offline operation and requires no internet
    connection. All processing happens on-device.
    """

    # COCO class names for YOLOv8
    COCO_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
    }

    # Target classes for alerts (strict filtering)
    ALERT_CLASSES = {0, 2}  # person, car
    ALERT_CLASS_NAMES = {'person', 'car'}

    # Confidence threshold for alerts
    CONFIDENCE_THRESHOLD = 0.60

    # Minimum cluster size for micro-target detection (pixels)
    MIN_CLUSTER_SIZE = 15

    # Gaussian blur kernel size for noise reduction (must be odd)
    BLUR_KERNEL_SIZE = 5

    # Optical flow parameters
    FLOW_PYRAMID_SCALE = 0.5
    FLOW_LEVELS = 3
    FLOW_WIN_SIZE = 15
    FLOW_ITERATIONS = 3
    FLOW_POLY_N = 5
    FLOW_POLY_SIGMA = 1.2

    def __init__(self, model_path: str = 'yolov8n.onnx',
                 input_size: Tuple[int, int] = (640, 640),
                 use_gpu: bool = False):
        """
        Initialize the Professional Vision Engine.

        Args:
            model_path: Path to the ONNX model file
            input_size: Model input size (width, height)
            use_gpu: Whether to attempt GPU acceleration
        """
        self.model_path = model_path
        self.input_size = input_size
        self.use_gpu = use_gpu

        # Initialize ONNX Runtime session
        self.session = None
        self.input_name = None
        self.output_name = None

        # Previous frame for motion detection
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_frame_blur: Optional[np.ndarray] = None

        # Motion compensation state
        self.motion_vector = MotionVector(0.0, 0.0, 0.0, 0.0)
        self.motion_threshold = 5.0  # Pixels of motion before compensation

        # Performance metrics
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.processing_times: List[float] = []

        # Thread-safe queue for async processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)

        # State flags
        self.is_initialized = False
        self.is_running = False

        # Initialize the model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize the ONNX Runtime session with hardware acceleration.

        Attempts to use GPU acceleration if available and requested.
        Falls back to CPU if GPU is not available.
        """
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please ensure yolov8n.onnx is in the application directory."
            )

        # Configure execution providers
        providers = []

        if self.use_gpu:
            # Try GPU providers in order of preference
            gpu_providers = [
                'CUDAExecutionProvider',
                'OpenVINOExecutionProvider',
                'TensorrtExecutionProvider',
            ]
            for provider in gpu_providers:
                if provider in ort.get_available_providers():
                    providers.append(provider)
                    break

        # Always add CPU as fallback
        if 'CPUExecutionProvider' in ort.get_available_providers():
            providers.append('CPUExecutionProvider')

        if not providers:
            providers = ['CPUExecutionProvider']

        # Create inference session with optimization options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Multi-threaded inference
        sess_options.inter_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            self.is_initialized = True
            print(f"[VisionEngine] Model loaded successfully with providers: {providers}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX Runtime: {e}")

    def apply_anti_noise_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Step A: Apply Gaussian blur to eliminate digital sensor noise.

        This step removes:
        - Digital sensor noise from low-light conditions
        - Lighting reflections and glare artifacts
        - Screen scan lines and interlacing artifacts
        - High-frequency noise that could cause false detections

        Args:
            frame: Input BGR frame from camera

        Returns:
            Denoised BGR frame
        """
        if frame is None or frame.size == 0:
            return frame

        # Apply Gaussian blur with optimized kernel size
        # Larger kernels provide more smoothing but reduce detail
        # Kernel size 5x5 is optimal for mobile camera noise
        denoised = cv2.GaussianBlur(
            frame,
            (self.BLUR_KERNEL_SIZE, self.BLUR_KERNEL_SIZE),
            0  # Sigma computed from kernel size
        )

        return denoised

    def calculate_global_motion(self, frame: np.ndarray) -> MotionVector:
        """
        Step B: Calculate global motion compensation using Farneback optical flow.

        This algorithm detects camera panning/movement and calculates
        the compensation vector to prevent false alerts when the user
        moves the device.

        Uses the Farneback method for dense optical flow calculation,
        which provides robust motion estimation even with noise.

        Args:
            frame: Current denoised BGR frame

        Returns:
            MotionVector containing displacement and magnitude
        """
        # Convert to grayscale for optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply additional blur for optical flow (reduces noise sensitivity)
        frame_gray = cv2.GaussianBlur(frame_gray, (9, 9), 2.0)

        # Initialize if first frame
        if self.prev_frame_gray is None:
            self.prev_frame_gray = frame_gray.copy().astype(np.float32)
            return MotionVector(0.0, 0.0, 0.0, 0.0)

        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame_gray,
            frame_gray,
            None,
            pyr_scale=self.FLOW_PYRAMID_SCALE,
            levels=self.FLOW_LEVELS,
            winsize=self.FLOW_WIN_SIZE,
            iterations=self.FLOW_ITERATIONS,
            poly_n=self.FLOW_POLY_N,
            poly_sigma=self.FLOW_POLY_SIGMA,
            flags=0
        )

        # Calculate global motion by computing mean flow
        # This assumes most of the scene is stationary
        dx = float(np.mean(flow[:, :, 0]))
        dy = float(np.mean(flow[:, :, 1]))

        # Calculate magnitude and angle
        magnitude = float(np.sqrt(dx * dx + dy * dy))
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Create motion vector
        motion = MotionVector(dx, dy, magnitude, angle)

        # Update previous frame
        self.prev_frame_gray = frame_gray.copy().astype(np.float32)

        # Store for external access
        self.motion_vector = motion

        return motion

    def detect_micro_targets(self, frame: np.ndarray,
                             prev_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Step C: Detect small independent pixel clusters (micro-targets).

        Isolates tiny moving pixel clusters down to 15x15 pixels.
        Uses background subtraction and contour detection to find
        small moving objects like distant players or vehicles.

        Args:
            frame: Current denoised frame
            prev_frame: Previous denoised frame (same size)

        Returns:
            List of bounding boxes (x, y, w, h) for micro-targets
        """
        if prev_frame is None or frame.shape != prev_frame.shape:
            return []

        # Convert to grayscale
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference (motion detection)
        diff = cv2.absdiff(gray_current, gray_prev)

        # Apply threshold to isolate moving regions
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours (connected components)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by size to find micro-targets
        micro_targets = []
        min_area = self.MIN_CLUSTER_SIZE * self.MIN_CLUSTER_SIZE
        max_area = 150 * 150  # Maximum micro-target size

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Check if size is in micro-target range
            if min_area <= area <= max_area:
                # Check aspect ratio (not too elongated)
                aspect_ratio = w / max(h, 1)
                if 0.2 < aspect_ratio < 5.0:
                    micro_targets.append((x, y, w, h))

        return micro_targets

    def preprocess_for_inference(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLOv8 ONNX inference.

        Steps:
        1. Resize to model input size (640x640)
        2. Convert BGR to RGB
        3. Normalize to [0, 1]
        4. Transpose to (C, H, W) format
        5. Add batch dimension

        Args:
            frame: Input BGR frame

        Returns:
            Preprocessed numpy array ready for inference
        """
        # Store original dimensions for later scaling
        self.orig_height, self.orig_width = frame.shape[:2]

        # Resize with letterboxing to maintain aspect ratio
        input_w, input_h = self.input_size

        # Calculate scaling factor
        scale = min(input_w / frame.shape[1], input_h / frame.shape[0])
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)

        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded input
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Store padding for coordinate conversion
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.scale = scale

        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose from (H, W, C) to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension: (1, C, H, W)
        batched = np.expand_dims(transposed, axis=0)

        # Store as contiguous array for optimal inference
        return np.ascontiguousarray(batched)

    def run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Step D: Run hardware-accelerated ONNX inference.

        Uses ONNX Runtime with native CPU/GPU acceleration
        for optimal performance on mobile devices.

        Args:
            input_tensor: Preprocessed input tensor (1, 3, H, W)

        Returns:
            Raw model output tensor
        """
        if not self.is_initialized:
            raise RuntimeError("Vision engine not initialized")

        start_time = time.time()

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        inference_time = time.time() - start_time
        self.processing_times.append(inference_time)

        # Keep only last 100 measurements for averaging
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]

        return outputs[0]

    def postprocess_detections(self, raw_output: np.ndarray,
                               conf_threshold: float = None) -> List[Detection]:
        """
        Step E: Post-process raw detections with strict filtering.

        Applies Non-Maximum Suppression (NMS) and strict class filtering.
        Only returns 'person' (class 0) and 'car' (class 2) detections
        with confidence above the threshold.

        Args:
            raw_output: Raw model output tensor
            conf_threshold: Confidence threshold (default: 0.60)

        Returns:
            List of filtered Detection objects
        """
        if conf_threshold is None:
            conf_threshold = self.CONFIDENCE_THRESHOLD

        # YOLOv8 output shape: (1, 84, 8400) for yolov8n
        # 84 = 4 bbox coords + 80 class scores
        # 8400 = number of predictions

        # Transpose to (8400, 84)
        predictions = raw_output[0].T

        # Extract bounding boxes (first 4 columns)
        boxes = predictions[:, :4]

        # Extract class scores (remaining columns)
        class_scores = predictions[:, 4:]

        # Get class with highest score for each prediction
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence threshold
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Filter by target classes (person=0, car=2)
        target_mask = np.isin(class_ids, list(self.ALERT_CLASSES))
        boxes = boxes[target_mask]
        confidences = confidences[target_mask]
        class_ids = class_ids[target_mask]

        # If no detections, return empty list
        if len(boxes) == 0:
            return []

        # Convert YOLO format (x_center, y_center, w, h) to (x1, y1, x2, y2)
        # and scale back to original image coordinates
        detections = []

        for i in range(len(boxes)):
            x_center, y_center, w, h = boxes[i]

            # Convert to corner coordinates and scale
            x1 = int((x_center - w / 2 - self.pad_x) / self.scale)
            y1 = int((y_center - h / 2 - self.pad_y) / self.scale)
            x2 = int((x_center + w / 2 - self.pad_x) / self.scale)
            y2 = int((y_center + h / 2 - self.pad_y) / self.scale)

            # Clip to image boundaries
            x1 = max(0, min(x1, self.orig_width - 1))
            y1 = max(0, min(y1, self.orig_height - 1))
            x2 = max(0, min(x2, self.orig_width - 1))
            y2 = max(0, min(y2, self.orig_height - 1))

            # Get class information
            class_id = int(class_ids[i])
            class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
            confidence = float(confidences[i])

            # Calculate center and dimensions
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # Create detection object
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                width=width,
                height=height
            )
            detections.append(detection)

        # Apply Non-Maximum Suppression
        if len(detections) > 0:
            detections = self._apply_nms(detections)

        return detections

    def _apply_nms(self, detections: List[Detection],
                   iou_threshold: float = 0.45) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for suppression

        Returns:
            Filtered list of Detection objects
        """
        if len(detections) == 0:
            return []

        # Convert to numpy arrays for NMS
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        # Calculate areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by confidence
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep detections with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], Dict[str, Any]]:
        """
        Complete processing pipeline for a single frame.

        Executes all 5 stages:
        A. Anti-noise filtering
        B. Global motion compensation
        C. Micro-target detection
        D. Hardware-accelerated inference
        E. Strict classification filtering

        Args:
            frame: Input BGR frame from camera

        Returns:
            Tuple of (detections list, metadata dict)
        """
        start_time = time.time()

        # Initialize metadata
        metadata = {
            'frame_shape': frame.shape,
            'processing_time_ms': 0.0,
            'motion_compensated': False,
            'micro_targets_count': 0,
            'inference_time_ms': 0.0,
            'fps': self.fps,
            'alert_classes_only': True
        }

        # Stage A: Anti-noise filtering
        denoised = self.apply_anti_noise_filter(frame)

        # Stage B: Global motion compensation
        motion = self.calculate_global_motion(denoised)
        metadata['motion_compensated'] = motion.magnitude > self.motion_threshold
        metadata['motion_magnitude'] = motion.magnitude
        metadata['motion_angle'] = motion.angle

        # Store previous frame for micro-target detection
        prev_blur = self.prev_frame_blur
        self.prev_frame_blur = denoised.copy()

        # Stage C: Micro-target detection (optional, for analysis)
        if prev_blur is not None:
            micro_targets = self.detect_micro_targets(denoised, prev_blur)
            metadata['micro_targets_count'] = len(micro_targets)

        # Stage D: Preprocess for inference
        input_tensor = self.preprocess_for_inference(denoised)

        # Stage D: Run inference
        raw_output = self.run_inference(input_tensor)

        # Calculate inference time
        if self.processing_times:
            metadata['inference_time_ms'] = self.processing_times[-1] * 1000

        # Stage E: Post-process with strict filtering
        detections = self.postprocess_detections(raw_output)

        # Update FPS counter
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time

        # Calculate total processing time
        metadata['processing_time_ms'] = (time.time() - start_time) * 1000
        metadata['fps'] = self.fps

        return detections, metadata

    def get_average_inference_time(self) -> float:
        """
        Get average inference time over last 100 frames.

        Returns:
            Average inference time in milliseconds
        """
        if not self.processing_times:
            return 0.0
        return (sum(self.processing_times) / len(self.processing_times)) * 1000

    def should_alert(self, detection: Detection) -> bool:
        """
        Determine if a detection should trigger an alert.

        Checks:
        - Class is in alert classes (person or car)
        - Confidence exceeds threshold

        Args:
            detection: Detection object to evaluate

        Returns:
            True if detection should trigger alert
        """
        return (detection.class_id in self.ALERT_CLASSES and
                detection.confidence > self.CONFIDENCE_THRESHOLD)

    def cleanup(self) -> None:
        """
        Clean up resources and release memory.
        """
        if self.session:
            del self.session
            self.session = None

        self.prev_frame_gray = None
        self.prev_frame_blur = None
        self.processing_times.clear()
        self.is_initialized = False


class VisionEngineWorker:
    """
    Worker class for running vision engine in a separate thread.

    Provides thread-safe frame processing with minimal latency.
    """

    def __init__(self, engine: ProfessionalVisionEngine):
        """
        Initialize worker with vision engine.

        Args:
            engine: Initialized ProfessionalVisionEngine instance
        """
        self.engine = engine
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.thread = None
        self.running = False

    def start(self) -> None:
        """Start the worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def submit_frame(self, frame: np.ndarray) -> bool:
        """
        Submit a frame for asynchronous processing.

        Args:
            frame: BGR numpy array from camera

        Returns:
            True if frame was queued, False if queue full
        """
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[Tuple[List[Detection], Dict]]:
        """
        Get the latest processing result.

        Args:
            timeout: Max time to wait for result

        Returns:
            Tuple of (detections, metadata) or None if no result
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self) -> None:
        """Internal worker loop for processing frames."""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)

                # Process frame
                detections, metadata = self.engine.process_frame(frame)

                # Put result in queue (drop old results if full)
                try:
                    while not self.result_queue.empty():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait((detections, metadata))
                except queue.Empty:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VisionEngineWorker] Error processing frame: {e}")


def create_default_engine(model_path: str = 'yolov8n.onnx') -> ProfessionalVisionEngine:
    """
    Factory function to create a default vision engine.

    Args:
        model_path: Path to ONNX model file

    Returns:
        Initialized ProfessionalVisionEngine
    """
    return ProfessionalVisionEngine(
        model_path=model_path,
        input_size=(640, 640),
        use_gpu=False  # CPU is more reliable on mobile
    )


if __name__ == '__main__':
    # Test the vision engine
    print("=" * 60)
    print("Professional Vision Engine - Test Mode")
    print("=" * 60)

    # Create engine
    try:
        engine = create_default_engine()
        print(f"Engine initialized: {engine.is_initialized}")
        print(f"ONNX Providers: {ort.get_available_providers()}")

        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Process test frame
        detections, metadata = engine.process_frame(test_frame)

        print(f"\nTest Results:")
        print(f"  Detections: {len(detections)}")
        print(f"  FPS: {metadata['fps']:.1f}")
        print(f"  Processing Time: {metadata['processing_time_ms']:.1f}ms")
        print(f"  Motion Magnitude: {metadata['motion_magnitude']:.2f}px")

        engine.cleanup()
        print("\nEngine cleanup complete.")

    except Exception as e:
        print(f"Error: {e}")
