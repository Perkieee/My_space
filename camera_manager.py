# camera_manager.py
import threading
import time
import logging
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    from picamera2.controls import Controls
except Exception:
    Picamera2 = None  # fallback to None if library not available

from crop_manager import CropBatchManager
from IMX500Wrapper import imx_wrapper
# Note: IMX500Wrapper and CropBatchManager are not imported here to keep this
# module testable and decoupled — they are passed into the CameraManager at init.

class CameraManager:
    """
    CameraManager
    -------------
    Responsibilities:
    - Initialize Picamera2 preview stream and start capture.
    - In a background thread: capture requests (frame + metadata), call IMX500Wrapper to
      extract detections from metadata, hand the frame + detections to CropBatchManager,
      and store frame-level metadata (timestamp + raw metadata) for later lookup.
    - Provide start()/stop() lifecycle and a small API to query frame metadata via frame index.
    """

    def __init__(
        self,
        imx_wrapper,                    # instance of IMX500Wrapper (must implement get_detections_from_metadata)
        crop_batch_manager,             # instance of CropBatchManager (must implement submit_frame_detections)
        preview_size: Tuple[int, int] = (640, 360),
        show_preview: bool = False,
        warmup_seconds: float = 2.0,
        run_time: Optional[float] = None,   # if set, camera will stop after run_time seconds
        logger: Optional[logging.Logger] = None,
    ):
        self.imx = imx_wrapper
        self.crop_mgr = crop_batch_manager
        self.preview_size = preview_size
        self.show_preview = show_preview
        self.warmup_seconds = warmup_seconds
        self.run_time = run_time
        self.logger = logger or logging.getLogger("CameraManager")

        # Camera and thread control
        self.picam2 = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Frame index counter (monotonic)
        self._frame_index = 0
        # Map frame_idx -> frame-level metadata (contains at least 'frame_timestamp' per your request)
        self._frame_metadata: Dict[int, Dict[str, Any]] = {}
        # Lock protecting access to frame metadata
        self._meta_lock = threading.Lock()

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        """
        Start Picamera2 and the capture thread.
        Raises RuntimeError if Picamera2 is not available.
        """
        if Picamera2 is None:
            raise RuntimeError("Picamera2 not available in this environment.")

        if self.picam2 is not None:
            self.logger.warning("Camera already started")
            return

        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": self.preview_size, "format": "RGB888"})
        self.picam2.configure(config)

        try:
            self.picam2.start()
        except Exception as e:
            # Clean up if start fails
            self.picam2 = None
            raise RuntimeError(f"Failed to start Picamera2: {e}") from e

        # Warmup AWB/AE if requested
        if self.warmup_seconds and self.warmup_seconds > 0:
            self.logger.info("Camera started: warming up AWB/AE for %.1f seconds...", float(self.warmup_seconds))
            time.sleep(float(self.warmup_seconds))

        # Reset control flags and start worker thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        self.logger.info("Camera capture thread started.")

    def stop(self, join_timeout: float = 2.0) -> None:
        """
        Signal stop and wait for thread + camera to shut down.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                self.logger.warning("Camera thread did not exit within timeout.")
            self._thread = None

        # stop camera if running
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception as e:
                self.logger.exception("Error stopping Picamera2: %s", e)
            finally:
                self.picam2 = None

        self.logger.info("CameraManager stopped.")

    # ---------------- metadata API ----------------
    def get_frame_metadata(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        Return a copy of the stored frame metadata (which includes 'frame_timestamp') for the given frame index.
        Returns None if not found.
        """
        with self._meta_lock:
            md = self._frame_metadata.get(frame_idx)
            return dict(md) if md is not None else None

    # ---------------- internal capture loop ----------------
    def _capture_loop(self) -> None:
        """
        Main capture loop running in background thread.

        Flow per loop:
         - capture request (req)
         - obtain frame array (req.make_array('main') or fallback capture_array)
         - extract metadata via req.get_metadata()
         - store frame-level metadata (frame_timestamp if present)
         - call imx.get_detections_from_metadata(metadata)
         - hand detections + frame to crop_batch_manager.submit_frame_detections(frame_idx, frame_rgb, detections)
         - optionally draw preview and show using OpenCV
         - release the request (req.release())
        """
        start_time = time.perf_counter()
        last_fps_time = time.perf_counter()
        frame_counter = 0
        live_fps = 0.0

        while not self._stop_event.is_set():
            # Optionally exit after run_time (if provided)
            if self.run_time is not None and (time.perf_counter() - start_time) >= float(self.run_time):
                self.logger.info("CameraManager run_time reached; exiting capture loop.")
                break

            req = None
            try:
                # Capture a request (blocks until available from camera pipeline)
                req = self.picam2.capture_request()
                try:
                    frame_rgb = req.make_array("main")
                except Exception:
                    # Fallback API
                    frame_rgb = self.picam2.capture_array()

                # Obtain metadata from the same request so boxes match frame exactly
                raw_metadata = req.get_metadata() or {}
                # Determine a reasonable frame timestamp: prefer 'SensorTimestamp', fallback to perf_counter/time
                frame_timestamp = raw_metadata.get("SensorTimestamp") or raw_metadata.get("Timestamp") or time.time()

                # Record frame metadata for downstream lookups
                frame_idx = self._frame_index
                with self._meta_lock:
                    self._frame_metadata[frame_idx] = {
                        "frame_timestamp": frame_timestamp,
                        "raw_metadata": raw_metadata,
                    }

                # --------------- IMX500 inference (postprocessing) ---------------
                try:
                    detections, imx_inf_ms, imx_post_ms = self.imx.get_detections_from_metadata(raw_metadata)
                except Exception as e:
                    self.logger.exception("IMX500Wrapper raised exception while getting detections: %s", e)
                    detections, imx_inf_ms, imx_post_ms = [], 0.0, 0.0

                # --------------- Submit detections to CropBatchManager ---------------
                try:
                    # crop_mgr will handle cropping/resizing asynchronously
                    self.crop_mgr.submit_frame_detections(frame_idx, frame_rgb, detections)
                except Exception as e:
                    self.logger.exception("CropBatchManager.submit_frame_detections failed: %s", e)

                # --------------- Optional preview display ---------------
                if self.show_preview:
                    try:
                        # Draw boxes for quick visual debugging (non-blocking, lightweight)
                        disp = frame_rgb.copy()
                        H, W = disp.shape[:2]
                        for (xmin, ymin, xmax, ymax, score) in detections:
                            x1 = int(xmin * W); y1 = int(ymin * H)
                            x2 = int(xmax * W); y2 = int(ymax * H)
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{score:.2f}"
                            cv2.putText(disp, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        # simple FPS overlay
                        frame_counter += 1
                        now = time.perf_counter()
                        if (now - last_fps_time) >= 1.0:
                            live_fps = frame_counter / (now - last_fps_time)
                            last_fps_time = now
                            frame_counter = 0
                        cv2.putText(disp, f"FPS: {live_fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        cv2.imshow("Camera Preview", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == 27:
                            self.logger.info("Preview ESC pressed - requesting stop.")
                            self._stop_event.set()
                    except Exception:
                        # Ensure preview errors don't kill capture loop
                        self.logger.exception("Preview draw failed.")
                # release request buffers quickly so libcamera can reuse them
                try:
                    if req:
                        req.release()
                except Exception:
                    # ignore release errors
                    pass

                # Advance frame index (monotonic)
                self._frame_index += 1

            except Exception as e:
                # Broad exception to avoid capture thread death; log and continue or break if fatal
                self.logger.exception("Capture loop exception: %s", e)
                # Attempt to release request if allocated
                try:
                    if req:
                        req.release()
                except Exception:
                    pass
                # Brief sleep to avoid tight error loop
                time.sleep(0.01)
                continue

        # capture loop finished; clean up preview windows if used
        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        self.logger.info("Camera capture thread exiting.")

