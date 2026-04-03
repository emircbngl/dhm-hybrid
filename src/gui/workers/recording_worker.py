from PySide6.QtCore import QThread, Signal
from pathlib import Path
import numpy as np
import tifffile
import queue
import time
import os

class RecordingWorker(QThread):
    """
    Background worker managing Video, Timelapse, and Scheduled execution cycles.
    Accepts arrays passed via thread-safe queues and processes disk IO accordingly.
    """
    error_occurred = Signal(str)
    frame_written = Signal(int)
    finished_recording = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_queue = queue.Queue()
        self._is_recording = False
        self._writer = None
        self._state = {}

    def setup(self, out_dir: Path, template: str, fmt: str, channel: str, mode: str, 
              vid_max_frames: int, tl_interval: float, tl_duration: int, sch_start_delay: int):
        self._state = {
            "out_dir": out_dir,
            "template": template,
            "format": fmt,
            "channel": channel,
            "mode": mode,
            "vid_max_frames": vid_max_frames,
            "tl_interval": tl_interval,
            "tl_duration": tl_duration,
            "sch_start_delay": sch_start_delay
        }
        self.frames_written = 0
        self._last_tl_time = 0.0
        self._tl_start_time = 0.0

    def push_frame(self, frame_dict: dict):
        """Called defensively from the main thread. frame_dict contains specific channels."""
        if not self._is_recording:
            return
            
        mode = self._state["mode"]
        if mode == "Manual Video":
            max_f = self._state["vid_max_frames"]
            if max_f > 0 and (self.frames_written + self.frame_queue.qsize() >= max_f):
                return
            self.frame_queue.put(frame_dict)
            
        elif mode == "Timelapse":
            now = time.time()
            if now - self._last_tl_time >= self._state["tl_interval"]:
                self.frame_queue.put(frame_dict)
                self._last_tl_time = now

    def stop_recording(self):
        self._is_recording = False

    def is_recording(self) -> bool:
        return self._is_recording

    def run(self):
        mode = self._state["mode"]
        
        if mode == "Scheduled":
            import time
            delay_sec = self._state["sch_start_delay"] * 60
            time.sleep(delay_sec)
            self._state["mode"] = "Timelapse"   # Fallback into timelapse mechanics
            mode = "Timelapse"

        self._is_recording = True
        self.frames_written = 0
        self._writer = None
        
        self._tl_start_time = time.time()
        self._last_tl_time = 0.0
        
        import cv2

        os.makedirs(self._state["out_dir"], exist_ok=True)
        
        while self._is_recording or not self.frame_queue.empty():
            if mode == "Timelapse":
                dur_sec = self._state["tl_duration"] * 60
                if dur_sec > 0 and (time.time() - self._tl_start_time >= dur_sec):
                    self._is_recording = False
            
            try:
                # short timeout to check loop invariants
                frame_dict = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            try:
                self._write_frame(frame_dict, self.frames_written)
                self.frames_written += 1
                self.frame_written.emit(self.frames_written)
                
                if mode == "Manual Video":
                    max_f = self._state["vid_max_frames"]
                    if max_f > 0 and self.frames_written >= max_f:
                        self._is_recording = False
                        
            except Exception as e:
                self.error_occurred.emit(f"Recording error: {e}")
                self._is_recording = False
                break
                
        self._cleanup_writer()
        self.finished_recording.emit(str(self._state["out_dir"]))
        
    def _write_frame(self, frame_dict: dict, idx: int):
        import cv2
        channel = self._state["channel"]
        
        if channel == "Amplitude only":
            disp = frame_dict.get("amplitude", np.zeros((10,10), dtype=np.uint8))
            cname = "amplitude"
        elif channel == "Phase only":
            disp = frame_dict.get("phase", np.zeros((10,10), dtype=np.uint8))
            cname = "phase"
        elif channel == "Input only":
            disp = frame_dict.get("input", np.zeros((10,10), dtype=np.uint8))
            cname = "input"
        else:
            # For simplicity, fallback to amplitude
            disp = frame_dict.get("amplitude", np.zeros((10,10), dtype=np.uint8))
            cname = "all"
            
        z_val = frame_dict.get("z_mm", 0.0)
        template = str(self._state["template"])
        filename = template.format(name="run", z=z_val, channel=cname, frame=idx)

        fmt = str(self._state["format"])
        out_dir = Path(self._state["out_dir"])
        path_base = out_dir / filename

        if fmt == "Individual PNGs":
            path = path_base.with_suffix(".png")
            disp_u8 = self._to_uint8(disp)
            cv2.imwrite(str(path), disp_u8)
            
        elif fmt == "TIFF Stack":
            # Strip frame count if stacking into a single file
            stack_name = template.format(name="run", z=z_val, channel=cname, frame=0)
            path = out_dir / f"{stack_name}.tiff"
            tifffile.imwrite(str(path), disp, append=True)
            
        elif "MP4" in fmt or "AVI" in fmt:
            if self._writer is None:
                ext = ".mp4" if "MP4" in fmt else ".avi"
                stack_name = template.format(name="run", z=z_val, channel=cname, frame=0)
                path = out_dir / f"{stack_name}{ext}"
                try:
                    h, w = disp.shape[:2]
                except AttributeError:
                    h, w = 10, 10
                fps = 30.0 if self._state["mode"] == "Manual Video" else 10.0
                is_color = getattr(disp, "ndim", 2) == 3
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') if "MP4" in fmt else cv2.VideoWriter_fourcc(*'MJPG')
                self._writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h), is_color)
                
            disp_u8 = self._to_uint8(disp)
            if getattr(disp_u8, "ndim", 2) == 2:
                disp_u8 = cv2.cvtColor(disp_u8, cv2.COLOR_GRAY2BGR)
            if self._writer is not None:
                self._writer.write(disp_u8)

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        arr_norm = arr - arr.min()
        max_val = arr_norm.max()
        if max_val > 0:
            arr_norm = arr_norm / max_val
        return np.clip(arr_norm * 255.0, 0, 255).astype(np.uint8)

    def _cleanup_writer(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
