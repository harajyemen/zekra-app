"""
  Zekra AI - خدمة الخلفية المحمية (Foreground Service)
  ======================================================
  خدمة أندرويد محمية تعمل بشكل مستمر في الخلفية مع إشعار دائم
  تضمن استمرار معالجة الذكاء الاصطناعي طوال فترة اللعب بدون انقطاع.

  Author: Zekra AI Team
  Version: 2.0.0
  """

  import os
  import sys
  import time
  import threading

  print("[ZekraService] ==============================")
  print("[ZekraService] Zekra AI Background Service v2.0")
  print("[ZekraService] ==============================")

  # ===== إعداد خدمة الأندرويد الأمامية =====
  service_instance = None
  notification_manager = None
  _service_running = True

  def _setup_foreground_service():
      """
      تحويل الخدمة إلى Foreground Service رسمية مع إشعار دائم.
      هذا يمنع أندرويد من إيقاف الخدمة عند انخفاض الذاكرة أو أثناء اللعب.
      """
      global service_instance, notification_manager

      try:
          from jnius import autoclass, cast

          # ===== استدعاء واجهات الأندرويد الرسمية =====
          PythonService      = autoclass('org.kivy.android.PythonService')
          NotificationBuilder = autoclass('android.app.Notification$Builder')
          NotificationManager = autoclass('android.app.NotificationManager')
          NotificationChannel = autoclass('android.app.NotificationChannel')
          PendingIntent       = autoclass('android.app.PendingIntent')
          Intent              = autoclass('android.content.Intent')
          Color               = autoclass('android.graphics.Color')
          Build               = autoclass('android.os.Build')
          Context             = autoclass('android.content.Context')

          service_instance = PythonService.mService
          channel_id       = "zekra_foreground_channel"
          channel_name     = "Zekra AI - معالجة الذكاء الاصطناعي"

          # ===== إنشاء قناة الإشعارات (مطلوبة من API 26+) =====
          notification_manager = service_instance.getSystemService(Context.NOTIFICATION_SERVICE)

          channel = NotificationChannel(
              channel_id,
              channel_name,
              NotificationManager.IMPORTANCE_LOW   # منخفض = لا صوت ولا اهتزاز
          )
          channel.setDescription("Zekra AI تعالج الشاشة وترسم طبقة التتبع")
          channel.setShowBadge(False)
          notification_manager.createNotificationChannel(channel)

          # ===== بناء الإشعار الدائم =====
          builder = NotificationBuilder(service_instance, channel_id)
          builder.setContentTitle("Zekra AI — نشط")
          builder.setContentText("جارٍ التتبع البصري الذكي…")
          builder.setSmallIcon(service_instance.getApplicationInfo().icon)
          builder.setOngoing(True)           # إشعار دائم لا يمكن إغلاقه
          builder.setPriority(-1)            # PRIORITY_LOW
          builder.setColor(Color.parseColor("#00E5FF"))  # أزرق فوسفوري

          notification = builder.build()
          service_instance.startForeground(1001, notification)

          print("[ZekraService] ✓ Foreground service active — notification shown")
          return True

      except Exception as e:
          print(f"[ZekraService] Non-Android environment or setup error: {e}")
          return False


  def _update_notification(status_text: str, fps: float = 0.0):
      """تحديث نص الإشعار الدائم بمعلومات الحالة الحالية."""
      global service_instance, notification_manager
      try:
          if service_instance is None:
              return

          from jnius import autoclass
          NotificationBuilder = autoclass('android.app.Notification$Builder')
          Color               = autoclass('android.graphics.Color')

          channel_id = "zekra_foreground_channel"
          builder = NotificationBuilder(service_instance, channel_id)
          builder.setContentTitle("Zekra AI — نشط")
          builder.setContentText(f"{status_text} | {fps:.1f} FPS")
          builder.setSmallIcon(service_instance.getApplicationInfo().icon)
          builder.setOngoing(True)
          builder.setPriority(-1)
          builder.setColor(Color.parseColor("#00E5FF"))

          notification_manager.notify(1001, builder.build())

      except Exception:
          pass   # تحديث الإشعار ليس حرجاً — تجاهل أي خطأ


  def _heartbeat_loop():
      """
      حلقة نبضات القلب — تُبقي الخدمة حية وتحدّث الإشعار كل 30 ثانية.
      مصممة لتكون خفيفة جداً على المعالج والبطارية.
      """
      global _service_running
      tick = 0
      last_fps_log = time.time()

      print("[ZekraService] Heartbeat loop started")

      while _service_running:
          try:
              time.sleep(2)
              tick += 1

              # تحديث الإشعار كل 30 ثانية (15 دورة × 2 ثانية)
              if tick % 15 == 0:
                  elapsed = time.time() - last_fps_log
                  _update_notification("جارٍ التتبع البصري الذكي", 0.0)
                  print(f"[ZekraService] ♥ Heartbeat tick={tick}")

          except KeyboardInterrupt:
              print("[ZekraService] Interrupted — shutting down gracefully")
              _service_running = False
              break
          except Exception as loop_err:
              # تجاهل أي خطأ داخلي ومتابعة الحلقة لمنع إيقاف الخدمة
              print(f"[ZekraService] Loop error bypassed: {loop_err}")
              time.sleep(5)

      print("[ZekraService] Heartbeat loop exited cleanly")


  def main():
      """نقطة الدخول الرئيسية للخدمة."""
      print("[ZekraService] Starting main service sequence...")

      # الخطوة 1: إعداد الـ Foreground Service
      _setup_foreground_service()

      # الخطوة 2: بدء حلقة نبضات القلب
      heartbeat_thread = threading.Thread(
          target=_heartbeat_loop,
          name="ZekraHeartbeat",
          daemon=True
      )
      heartbeat_thread.start()

      print("[ZekraService] All threads running — service is fully operational")

      # الخطوة 3: إبقاء العملية الرئيسية حية
      try:
          heartbeat_thread.join()
      except Exception as fatal:
          print(f"[ZekraService] Fatal error bypassed — service continues: {fatal}")


  if __name__ == '__main__':
      main()
  