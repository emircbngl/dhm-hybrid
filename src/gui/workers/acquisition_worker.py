from PySide6.QtCore import QThread, Signal
import time
import numpy as np
from core.camera import NICamera, CameraError

class AcquisitionWorker(QThread):
    """
    Dedicated thread for camera frame grabbing.
    Runs a tight loop: grab frame -> emit signal -> optional FPS throttle.
    """
    frame_ready = Signal(object, int)      # (np.ndarray, frame_number)
    fps_updated = Signal(float)            # actual measured FPS
    error_occurred = Signal(str)           # error message
    acquisition_started = Signal()
    acquisition_stopped = Signal()

    def __init__(self, camera: NICamera):
        super().__init__()
        self.camera = camera
        self.running = False
        self.target_fps = 30        # 0 = unlimited
        self.frame_number = 0

    def run(self):
        self.running = True
        self.frame_number = 0
        self.acquisition_started.emit()

        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        fps_counter = 0
        fps_timer_start = time.perf_counter()

        while self.running:
            loop_start = time.perf_counter()
            try:
                frame = self.camera.grab_frame()
                self.frame_number += 1
                self.frame_ready.emit(frame, self.frame_number)

                # FPS measurement
                fps_counter += 1
                elapsed = time.perf_counter() - fps_timer_start
                if elapsed >= 1.0:
                    self.fps_updated.emit(fps_counter / elapsed)
                    fps_counter = 0
                    fps_timer_start = time.perf_counter()

                # FPS throttle
                if frame_interval > 0:
                    sleep_time = frame_interval - (time.perf_counter() - loop_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except CameraError as e:
                self.error_occurred.emit(str(e))
                break
            except Exception as e:
                self.error_occurred.emit(f"Unknown acquisition error: {e}")
                break

        self.acquisition_stopped.emit()

    def stop(self):
        self.running = False
