"""
  Zekra AI - محرك الرؤية الذكية الاحترافي
  ==========================================
  محرك رؤية حاسوبية عالي الأداء يعمل بالكامل بدون إنترنت.
  يتضمن: تتبع كالمان التنبؤي، تعزيز الصورة، استنتاج ONNX.

  Author: Zekra AI Team
  Version: 2.0.0
  """

  import numpy as np
  import threading
  import time
  import os
  import math
  from dataclasses import dataclass, field
  from typing import List, Tuple, Optional, Dict

  # ===== استيراد آمن للمكتبات الاختيارية =====
  try:
      import cv2
      CV2_AVAILABLE = True
  except ImportError:
      CV2_AVAILABLE = False
      print("[ZekraEngine] OpenCV not available — using numpy fallback")

  try:
      import onnxruntime as ort
      ORT_AVAILABLE = True
  except ImportError:
      ORT_AVAILABLE = False
      print("[ZekraEngine] ONNX Runtime not available — demo mode active")


  # ===== هياكل البيانات =====

  @dataclass
  class Detection:
      """كشف مفرد من نموذج الذكاء الاصطناعي."""
      class_id:    int
      class_name:  str
      confidence:  float
      bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2)

      @property
      def center(self) -> Tuple[int, int]:
          return ((self.bbox[0] + self.bbox[2]) // 2,
                  (self.bbox[1] + self.bbox[3]) // 2)

      @property
      def width(self) -> int:
          return self.bbox[2] - self.bbox[0]

      @property
      def height(self) -> int:
          return self.bbox[3] - self.bbox[1]

      def to_xywh(self) -> Tuple[float, float, float, float]:
          """تحويل إلى صيغة (cx, cy, w, h) للكالمان فلتر."""
          cx = (self.bbox[0] + self.bbox[2]) / 2.0
          cy = (self.bbox[1] + self.bbox[3]) / 2.0
          w  = float(self.width)
          h  = float(self.height)
          return cx, cy, w, h


  @dataclass
  class TrackedObject:
      """جسم مُتتبَّع مع حالة كالمان والمعلومات البصرية."""
      track_id:         int
      class_name:       str
      confidence:       float
      bbox:             Tuple[int, int, int, int]   # آخر إطار مُشاهَد / متوقَّع
      frames_since_seen: int = 0
      age:              int = 0                     # عمر المسار بالإطارات
      is_predicted:     bool = False               # صحيح إذا كان الموضع متوقَّعاً


  # ===== كالمان فلتر لتتبع جسم واحد =====

  class KalmanTracker:
      """
      كالمان فلتر ثماني الأبعاد لتتبع الأجسام المتحركة.
      الحالة: [cx, cy, w, h, vcx, vcy, vw, vh]
        cx, cy = مركز الإطار المحيط
        w,  h  = عرض وارتفاع الإطار
        vcx,vcy= سرعة المركز
        vw, vh = معدل تغير الحجم
      القياس: [cx, cy, w, h]
      """

      def __init__(self, initial_bbox: Tuple[int, int, int, int]):
          x1, y1, x2, y2 = initial_bbox
          cx = (x1 + x2) / 2.0
          cy = (y1 + y2) / 2.0
          w  = float(x2 - x1)
          h  = float(y2 - y1)

          # ===== حالة الحركة (8D) =====
          self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float64).reshape(8, 1)

          # مصفوفة الانتقال F: x_k+1 = F @ x_k
          dt = 1.0
          self.F = np.eye(8, dtype=np.float64)
          self.F[0, 4] = dt   # cx += vcx
          self.F[1, 5] = dt   # cy += vcy
          self.F[2, 6] = dt   # w  += vw
          self.F[3, 7] = dt   # h  += vh

          # مصفوفة القياس H: z = H @ x
          self.H = np.zeros((4, 8), dtype=np.float64)
          self.H[0, 0] = 1.   # cx
          self.H[1, 1] = 1.   # cy
          self.H[2, 2] = 1.   # w
          self.H[3, 3] = 1.   # h

          # ضوضاء العملية Q — كيف نثق بنموذج الحركة
          self.Q = np.eye(8, dtype=np.float64)
          self.Q[0, 0] = 1.;   self.Q[1, 1] = 1.    # موقع
          self.Q[2, 2] = 10.;  self.Q[3, 3] = 10.   # حجم
          self.Q[4, 4] = 0.01; self.Q[5, 5] = 0.01  # سرعة موقع
          self.Q[6, 6] = 0.1;  self.Q[7, 7] = 0.1   # سرعة حجم

          # ضوضاء القياس R — كيف نثق بالكشف الجديد
          self.R = np.eye(4, dtype=np.float64)
          self.R[0, 0] = 1.;  self.R[1, 1] = 1.    # موقع دقيق
          self.R[2, 2] = 10.; self.R[3, 3] = 10.   # حجم أقل دقة

          # مصفوفة عدم اليقين P
          self.P = np.eye(8, dtype=np.float64) * 10.0

      # ------- خطوة التنبؤ (بدون قياس) -------
      def predict(self) -> Tuple[int, int, int, int]:
          """تقدير الموضع التالي بناءً على الحركة الحالية."""
          self.x = self.F @ self.x
          self.P = self.F @ self.P @ self.F.T + self.Q
          return self._state_to_bbox()

      # ------- خطوة التحديث (مع قياس جديد) -------
      def update(self, bbox: Tuple[int, int, int, int]):
          """تحديث الحالة بقياس جديد من الكاشف."""
          x1, y1, x2, y2 = bbox
          z = np.array([(x1+x2)/2., (y1+y2)/2.,
                        float(x2-x1), float(y2-y1)],
                       dtype=np.float64).reshape(4, 1)

          # كسب كالمان
          S = self.H @ self.P @ self.H.T + self.R
          K = self.P @ self.H.T @ np.linalg.inv(S)

          # تحديث الحالة وعدم اليقين
          self.x = self.x + K @ (z - self.H @ self.x)
          self.P = (np.eye(8) - K @ self.H) @ self.P

      def _state_to_bbox(self) -> Tuple[int, int, int, int]:
          """تحويل حالة الكالمان إلى إطار محيط (x1,y1,x2,y2)."""
          cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
          w = max(w, 1.0);  h = max(h, 1.0)
          return (int(cx - w/2), int(cy - h/2),
                  int(cx + w/2), int(cy + h/2))


  # ===== محدد تداخل الإطارات (IoU) =====

  def _iou(boxA: Tuple, boxB: Tuple) -> float:
      """حساب نسبة التقاطع إلى الاتحاد بين إطارين محيطين."""
      xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
      xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])

      inter = max(0, xB - xA) * max(0, yB - yA)
      if inter == 0:
          return 0.0

      areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
      areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
      return inter / float(areaA + areaB - inter)


  # ===== محرك الرؤية الرئيسي =====

  class ProfessionalVisionEngine:
      """
      محرك رؤية حاسوبية كامل يدمج:
      - تعزيز الصورة (زوم رقمي + تحسين التباين)
      - استنتاج ONNX بـ YOLOv8
      - تتبع متعدد الأجسام بكالمان فلتر
      - استمرار التتبع عند الاختفاء المؤقت
      """

      # ===== فئات COCO المدعومة =====
      COCO_CLASSES = {
          0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
          4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
          24: 'backpack', 26: 'handbag', 28: 'suitcase',
          32: 'sports ball', 39: 'bottle', 56: 'chair', 60: 'dining table',
          63: 'laptop', 67: 'cell phone', 73: 'book',
      }

      CONFIDENCE_THRESHOLD = 0.50    # 50% حد ثقة أدنى للكشف
      NMS_THRESHOLD        = 0.45    # حد حذف التكرارات
      MAX_LOST_FRAMES      = 15      # عدد إطارات إبقاء الجسم المتوقع بعد الاختفاء
      IOU_ASSIGNMENT_THRESHOLD = 0.30  # حد التطابق بين متتبع وكشف جديد

      def __init__(
          self,
          model_path:  str = 'yolov8n.onnx',
          input_size:  Tuple[int, int] = (640, 640),
          zoom_factor: float = 1.5,
      ):
          self.model_path  = model_path
          self.input_size  = input_size
          self.zoom_factor = max(1.0, zoom_factor)

          self._session    = None
          self._input_name = None
          self._trackers:  Dict[int, KalmanTracker] = {}
          self._tracked:   Dict[int, TrackedObject]  = {}
          self._next_id    = 0
          self._lock       = threading.Lock()

          self.frame_count = 0
          self.fps         = 0.0
          self._fps_time   = time.time()

          self._initialize_model()

      # ==================== تهيئة النموذج ====================

      def _initialize_model(self):
          """تحميل نموذج ONNX بأمان تام مع دعم GPU."""
          if not ORT_AVAILABLE:
              print("[ZekraEngine] Demo mode — ONNX Runtime missing")
              return

          if not os.path.exists(self.model_path):
              print(f"[ZekraEngine] Model not found: {self.model_path} — Demo mode")
              return

          try:
              providers = ['CPUExecutionProvider']
              if 'CUDAExecutionProvider' in ort.get_available_providers():
                  providers.insert(0, 'CUDAExecutionProvider')

              opts = ort.SessionOptions()
              opts.intra_op_num_threads  = 4
              opts.inter_op_num_threads  = 2
              opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

              self._session    = ort.InferenceSession(self.model_path,
                                                       sess_options=opts,
                                                       providers=providers)
              self._input_name = self._session.get_inputs()[0].name

              print(f"[ZekraEngine] ✓ Model loaded: {self.model_path}")
              print(f"[ZekraEngine] ✓ Providers: {self._session.get_providers()}")

          except Exception as e:
              print(f"[ZekraEngine] Model load failed — Demo mode: {e}")
              self._session = None

      # ==================== تعزيز الصورة ====================

      def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
          """
          تعزيز الصورة للأجسام البعيدة:
          1. زوم رقمي ذكي من المنتصف
          2. تحسين التباين بـ CLAHE (Contrast Limited Adaptive Histogram Equalization)
          """
          h, w = frame.shape[:2]

          # ----- خطوة 1: الزوم الرقمي -----
          if self.zoom_factor > 1.01:
              crop_h = int(h / self.zoom_factor)
              crop_w = int(w / self.zoom_factor)
              y0 = (h - crop_h) // 2
              x0 = (w - crop_w) // 2
              cropped = frame[y0:y0+crop_h, x0:x0+crop_w]

              if CV2_AVAILABLE:
                  frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
              else:
                  # numpy fallback — أبطأ لكن آمن
                  y_idx = (np.arange(h) * crop_h / h).astype(int)
                  x_idx = (np.arange(w) * crop_w / w).astype(int)
                  frame = cropped[np.ix_(y_idx, x_idx)]

          # ----- خطوة 2: تحسين التباين (CLAHE) -----
          if CV2_AVAILABLE and frame.ndim == 3:
              lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
              l, a, b = cv2.split(lab)
              clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
              l_eq  = clahe.apply(l)
              lab_eq = cv2.merge([l_eq, a, b])
              frame  = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

          return frame

      # ==================== المعالجة المسبقة لـ ONNX ====================

      def _preprocess(self, frame: np.ndarray) -> np.ndarray:
          """تحويل الإطار إلى tensor مناسب لـ YOLOv8."""
          if CV2_AVAILABLE:
              resized = cv2.resize(frame, self.input_size)
              rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
          else:
              # numpy fallback
              h, w = frame.shape[:2]
              th, tw = self.input_size[1], self.input_size[0]
              y_idx = (np.arange(th) * h / th).astype(int)
              x_idx = (np.arange(tw) * w / tw).astype(int)
              rgb   = frame[np.ix_(y_idx, x_idx)]
              if rgb.shape[2] == 4:
                  rgb = rgb[:, :, :3]

          tensor = rgb.astype(np.float32) / 255.0
          tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
          tensor = np.expand_dims(tensor, axis=0)   # -> NCHW
          return tensor

      # ==================== الاستنتاج وما بعده ====================

      def _run_inference(self, tensor: np.ndarray) -> List[Detection]:
          """تشغيل نموذج YOLOv8 وإرجاع قائمة الكشوفات."""
          if self._session is None:
              return self._demo_detections()

          try:
              outputs = self._session.run(None, {self._input_name: tensor})
              return self._postprocess(outputs)
          except Exception as e:
              print(f"[ZekraEngine] Inference error: {e}")
              return []

      def _postprocess(self, outputs: list) -> List[Detection]:
          """
          تفسير مخرجات YOLOv8.
          الشكل المتوقع: [1, 84, 8400] حيث:
            - 84 = 4 إحداثيات + 80 فئة
            - 8400 = عدد الـ anchors
          """
          preds  = outputs[0]   # (1, 84, 8400)
          preds  = np.squeeze(preds, axis=0).T  # -> (8400, 84)

          boxes  = preds[:, :4]   # cx, cy, w, h (مُعيَّر على input_size)
          scores = preds[:, 4:]   # (8400, 80)

          class_ids   = np.argmax(scores, axis=1)
          confidences = scores[np.arange(len(class_ids)), class_ids]

          # تصفية بحد الثقة
          mask   = confidences >= self.CONFIDENCE_THRESHOLD
          boxes  = boxes[mask];   confidences = confidences[mask];  class_ids = class_ids[mask]

          if len(boxes) == 0:
              return []

          # تحويل cx,cy,w,h -> x1,y1,x2,y2 على input_size
          cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
          x1 = cx - w/2;  y1 = cy - h/2
          x2 = cx + w/2;  y2 = cy + h/2

          # حذف التكرارات بـ NMS
          if CV2_AVAILABLE:
              nms_boxes  = np.stack([x1, y1, w, h], axis=1)
              indices    = cv2.dnn.NMSBoxes(
                  nms_boxes.tolist(),
                  confidences.tolist(),
                  self.CONFIDENCE_THRESHOLD,
                  self.NMS_THRESHOLD
              )
              if len(indices) == 0:
                  return []
              indices = indices.flatten()
          else:
              indices = list(range(len(confidences)))

          # تطبيع الإحداثيات إلى [0,1] ثم ضربها بحجم الإطار الفعلي لاحقاً
          iw, ih = self.input_size
          detections = []
          for i in indices:
              x1i = max(0, int(x1[i]));  y1i = max(0, int(y1[i]))
              x2i = min(iw, int(x2[i])); y2i = min(ih, int(y2[i]))
              cid = int(class_ids[i])
              name = self.COCO_CLASSES.get(cid, f'obj_{cid}')
              detections.append(Detection(
                  class_id=cid,
                  class_name=name,
                  confidence=float(confidences[i]),
                  bbox=(x1i, y1i, x2i, y2i)
              ))

          return detections

      def _demo_detections(self) -> List[Detection]:
          """كشوفات تجريبية عند غياب النموذج — لأغراض الاختبار."""
          t = time.time()
          cx = int(300 + 100 * math.sin(t * 0.8))
          cy = int(200 + 60  * math.cos(t * 0.6))
          return [Detection(class_id=0, class_name='person',
                            confidence=0.92,
                            bbox=(cx-40, cy-80, cx+40, cy+80))]

      # ==================== مطابقة المتتبعين بالكشوفات ====================

      def _scale_bbox(self, bbox, frame_w, frame_h):
          """تحويل إحداثيات input_size إلى إحداثيات الإطار الفعلي."""
          iw, ih = self.input_size
          sx, sy = frame_w / iw, frame_h / ih
          x1, y1, x2, y2 = bbox
          return (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))

      def _assign_and_update(
          self,
          detections: List[Detection],
          frame_w: int, frame_h: int
      ) -> List[TrackedObject]:
          """
          خوارزمية مطابقة جشعة (Greedy IoU Matching):
          1. تنبؤ مواضع جميع المتتبعين
          2. مطابقة كل كشف بأقرب متتبع (IoU)
          3. تحديث المتتبعين المتطابقين
          4. إنشاء متتبعين جدد للكشوفات غير المتطابقة
          5. حذف المتتبعين الذين تجاوزوا حد الإطارات الضائعة
          """
          with self._lock:
              # ----- خطوة 1: تنبؤ المواضع -----
              predicted = {}
              for tid, tracker in self._trackers.items():
                  predicted[tid] = tracker.predict()

              # ----- خطوة 2: مطابقة الكشوفات -----
              unmatched_dets = list(range(len(detections)))
              matched_trackers = set()

              for det_i in list(unmatched_dets):
                  det = detections[det_i]
                  scaled_bbox = self._scale_bbox(det.bbox, frame_w, frame_h)

                  best_iou, best_tid = 0.0, None
                  for tid, pred_bbox in predicted.items():
                      if tid in matched_trackers:
                          continue
                      iou_val = _iou(scaled_bbox, pred_bbox)
                      if iou_val > best_iou:
                          best_iou, best_tid = iou_val, tid

                  if best_tid is not None and best_iou >= self.IOU_ASSIGNMENT_THRESHOLD:
                      # تحديث المتتبع المتطابق
                      self._trackers[best_tid].update(scaled_bbox)
                      obj = self._tracked[best_tid]
                      obj.bbox             = self._trackers[best_tid]._state_to_bbox()
                      obj.confidence       = det.confidence
                      obj.frames_since_seen = 0
                      obj.is_predicted     = False
                      obj.age             += 1

                      matched_trackers.add(best_tid)
                      unmatched_dets.remove(det_i)

              # ----- خطوة 3: متتبعون جدد للكشوفات غير المتطابقة -----
              for det_i in unmatched_dets:
                  det = detections[det_i]
                  scaled_bbox = self._scale_bbox(det.bbox, frame_w, frame_h)
                  tid  = self._next_id
                  self._next_id += 1

                  self._trackers[tid] = KalmanTracker(scaled_bbox)
                  self._tracked[tid]  = TrackedObject(
                      track_id=tid,
                      class_name=det.class_name,
                      confidence=det.confidence,
                      bbox=scaled_bbox
                  )

              # ----- خطوة 4: تحديث المتتبعين الضائعين -----
              for tid in list(self._trackers.keys()):
                  if tid not in matched_trackers:
                      obj = self._tracked[tid]
                      obj.frames_since_seen += 1
                      obj.is_predicted       = True
                      obj.bbox               = predicted[tid]   # موضع كالمان المتوقَّع

                      if obj.frames_since_seen > self.MAX_LOST_FRAMES:
                          del self._trackers[tid]
                          del self._tracked[tid]

              return list(self._tracked.values())

      # ==================== الدالة الرئيسية: معالجة الإطار ====================

      def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
          """
          خط الأنابيب الكامل لمعالجة إطار واحد:
          1. تعزيز الصورة (زوم + CLAHE)
          2. المعالجة المسبقة للنموذج
          3. استنتاج ONNX
          4. تحديث المتتبعين بكالمان فلتر
          5. إرجاع قائمة الأجسام المتتبَّعة

          Args:
              frame: إطار numpy بصيغة BGR (من OpenCV أو MediaProjection)

          Returns:
              قائمة TrackedObject مع الإحداثيات والحالة
          """
          if frame is None or frame.size == 0:
              return []

          t0 = time.time()
          frame_h, frame_w = frame.shape[:2]

          try:
              # الخطوة 1: تعزيز الصورة
              enhanced = self._enhance_frame(frame)

              # الخطوة 2: معالجة مسبقة
              tensor = self._preprocess(enhanced)

              # الخطوة 3: استنتاج النموذج
              detections = self._run_inference(tensor)

              # الخطوة 4: تتبع كالمان
              tracked = self._assign_and_update(detections, frame_w, frame_h)

          except Exception as e:
              print(f"[ZekraEngine] process_frame error: {e}")
              tracked = []

          # ===== حساب FPS =====
          self.frame_count += 1
          elapsed = time.time() - self._fps_time
          if elapsed >= 1.0:
              self.fps       = self.frame_count / elapsed
              self.frame_count = 0
              self._fps_time   = time.time()

          return tracked
  