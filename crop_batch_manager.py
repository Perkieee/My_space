import time
import uuid
import cv2
import queue
import logging
import numpy as np
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque

class CropBatchManager:
    def __init__(self, cfg, hailo_input_shape, logger):
        self.cfg = cfg
        self.h_input_shape = hailo_input_shape
        self.logger = logger

        self.crop_pool = ThreadPoolExecutor(max_workers=cfg.crop_thread_workers)

        self.crop_result_queue = queue.Queue(
            maxsize=cfg.queue_maxsize * cfg.batch_max_size * 2
        )

        self.batch_queue = queue.Queue(maxsize=cfg.queue_maxsize)

        self.crop_timings = deque(maxlen=1000)
        self.batch_prepare_timings = deque(maxlen=1000)

        self._stop_event = False
        self._batch_thread = logging.Thread(
            target=self._batch_assembler_thread,
            daemon=True
        )
        self._batch_thread.start()

    def stop(self):
        self._stop_event = True
        self.crop_pool.shutdown(wait=False)

    @staticmethod
    def crop_to_bbox(image, bbox_norm):
        H, W = image.shape[:2]
        xmin_n, ymin_n, xmax_n, ymax_n = bbox_norm

        x1 = int(max(0, xmin_n * W))
        y1 = int(max(0, ymin_n * H))
        x2 = int(min(W, xmax_n * W))
        y2 = int(min(H, ymax_n * H))

        if x1 >= x2 or y1 >= y2:
            return np.zeros((1, 1, 3), dtype=image.dtype)

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def resize_for_hailo(img, target_shape):
        h, w = target_shape[:2]
        out = cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img

        if out.dtype != np.uint8:
            out = out.astype(np.uint8)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

        return out

    def submit_frame_detections(self, frame_idx, frame_rgb, detections):
        if not detections:
            return

        for det_idx, det in enumerate(detections):
            bbox = det[:4]
            submit_t = time.perf_counter()

            fut: Future = self.crop_pool.submit(
                self._crop_and_resize_task,
                frame_rgb,
                bbox,
                submit_t,
            )
            fut.add_done_callback(
                lambda f, fi=frame_idx, di=det_idx: self._on_crop_done(f, fi, di)
            )

    def _crop_and_resize_task(self, frame_rgb, bbox_norm, submit_time):
        t0 = time.perf_counter()
        roi = self.crop_to_bbox(frame_rgb, bbox_norm)
        roi_resized = self.resize_for_hailo(roi, self.h_input_shape)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        return roi_resized, bbox_norm, elapsed_ms

    def _on_crop_done(self, fut, frame_idx, det_idx):
        try:
            roi_resized, bbox_norm, crop_ms = fut.result()
        except Exception as e:
            self.logger.error("Crop task failed: %s", e)
            return

        item = (frame_idx, det_idx, roi_resized, bbox_norm, time.perf_counter(), crop_ms)

        try:
            self.crop_result_queue.put_nowait(item)
            self.crop_timings.append(crop_ms)
        except queue.Full:
            self.logger.warning("Crop queue full — dropping crop")

    def _batch_assembler_thread(self):
        partial = []
        last_time = time.perf_counter()

        while not self._stop_event:
            try:
                item = self.crop_result_queue.get(timeout=0.1)
            except queue.Empty:
                if partial and (time.perf_counter() - last_time) > 0.05:
                    self._flush(partial)
                    partial = []
                continue

            frame_idx, det_idx, roi_resized, bbox_norm, enqueue_time, crop_ms = item

            meta = {
                "frame_idx": frame_idx,
                "det_idx": det_idx,
                "bbox_norm": bbox_norm,
                "crop_enqueue_time": enqueue_time,
            }

            partial.append({"image": roi_resized, "meta": meta})

            if len(partial) >= self.cfg.batch_max_size:
                self._flush(partial)
                partial = []
                last_time = time.perf_counter()

        if partial:
            self._flush(partial)

    def _flush(self, items):
        t0 = time.perf_counter()
        batch_id = str(uuid.uuid4())

        frames = [x["image"] for x in items]
        metas = [x["meta"] for x in items]

        batch = {
            "batch_id": batch_id,
            "frames": frames,
            "metas": metas,
            "size": len(frames),
            "assembled_time": time.perf_counter(),
        }

        try:
            self.batch_queue.put_nowait(batch)
        except queue.Full:
            self.logger.warning("Batch queue full — dropping batch")

        t1 = time.perf_counter()
        self.batch_prepare_timings.append((t1 - t0) * 1000)

    def get_next_batch(self, timeout=0.1):
        try:
            return self.batch_queue.get(timeout=timeout)
        except queue.Empty:
            return None
