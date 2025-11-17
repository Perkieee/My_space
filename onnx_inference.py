import time
from abc import ABC, abstractmethod

import torch

from lib.inference.onnx_session import ONNXSession
from lib.inference.preprocessor import Preprocessor
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
import threading
from queue import Queue, Empty

WARMUP_ITERS = 5     # run this many warmup batches (ignored in timing)


class  AbstractSessionManager(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def reset_metrics(self):
        pass

    @abstractmethod
    def get_performance_metrics(self)->Dict[str,float]:
        pass

    @abstractmethod
    def infer(
            self,
            id_img_list: List[Tuple[str, np.ndarray]],
            multilabel_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        pass

class ONNXInference(AbstractSessionManager):

    def __init__(self,onnx_model_path:str):
        self._total_batches = None
        self._total_processing_ms = None
        self._total_decode_result_ms = None
        self._total_inference_ms = None
        self._total_preprocessing_ms = None
        self._total_images_processed = None
        self.onnx_model_path = onnx_model_path
        self.session = ONNXSession(self.onnx_model_path)
        self.preprocessors = None
        self.logger = logging.getLogger(__name__)
        self.warmup_iters = WARMUP_ITERS
        self.warm_left=self.warmup_iters
        self.performance_metrics={}
        # Async metrics update infrastructure
        self._metrics_queue: Queue = Queue()
        self._metrics_thread: Optional[threading.Thread] = None
        self._metrics_stop_event = threading.Event()
        self._metrics_lock = threading.Lock()

    def initialize(self):
        try:
            self.session.initialize()
            self.preprocessors= Preprocessor(pre_process_function_name=self.session.preprocessing)
            self.reset_metrics()
            # Ensure metrics worker is running
            self._start_metrics_worker()

        except Exception as e:
            self.logger.error(f"Failed to initialize ONNX session: {e}")
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")

    def __del__(self):
        # Best-effort to stop background worker when object is destroyed
        try:
            self.stop_metrics_worker(wait=False)
        except Exception:
            pass

    def reset_metrics(self):
        with self._metrics_lock:
            self.performance_metrics={
                "preprocessing_mean_ms": 0.0, # Average time spent on preprocessing by image (milliseconds)
                "inference_mean_ms": 0.0,     # Average time spent on inference by image (milliseconds)
                "decode_result_mean_ms": 0.0, # Average time spent on decoding results by image (milliseconds)
                "total_processing_mean_ms": 0.0,   # Average time spent on preprocessing, inference and decoding results by image (milliseconds)
                "throughput_images_per_sec": 0.0, # Number of images processed per second
            }
            # Internal accumulators for running metrics (post-warmup only)
            self._total_images_processed = 0
            self._total_preprocessing_ms = 0.0
            self._total_inference_ms = 0.0
            self._total_decode_result_ms = 0.0
            self._total_processing_ms = 0.0
            self._total_batches = 0

    def update_metrics(self,batch_size:int,metrics:Dict[str,float]):
        """
        Enqueue performance metrics to be processed asynchronously by a background worker.
        metrics keys expected (milliseconds for the whole batch):
          - preprocessing_ms
          - inference_ms
          - decode_result_ms
        batch_size is the number of images in the batch.
        """
        try:
            if batch_size is None or batch_size <= 0:
                return
            # Ensure worker is running
            self._start_metrics_worker()
            # Non-blocking put to the queue
            self._metrics_queue.put((int(batch_size), dict(metrics)))
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Failed to enqueue metrics: {e}")

    def _start_metrics_worker(self):
        if self._metrics_thread is None or not self._metrics_thread.is_alive():
            # Clear any previous stop signal and start a new worker
            self._metrics_stop_event.clear()
            self._metrics_thread = threading.Thread(target=self._metrics_worker, name="onnx_metrics_worker", daemon=True)
            self._metrics_thread.start()

    def stop_metrics_worker(self, wait: bool = False):
        """Signal the metrics worker to stop. Optionally wait for it to finish."""
        try:
            self._metrics_stop_event.set()
            if wait and self._metrics_thread is not None:
                self._metrics_thread.join(timeout=2.0)
        except Exception:
            pass

    def _metrics_worker(self):
        """Background thread that consumes metric updates and applies them."""
        while not self._metrics_stop_event.is_set() or not self._metrics_queue.empty():
            try:
                batch_size, metrics = self._metrics_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._apply_metrics(batch_size, metrics)
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"Metrics worker failed applying metrics: {e}")
            finally:
                self._metrics_queue.task_done()

    def _apply_metrics(self, batch_size: int, metrics: Dict[str, float]):
        """Apply metrics to internal accumulators (original synchronous logic)."""
        try:
            prep_ms = float(metrics.get("preprocessing_ms", 0.0) or 0.0)
            inf_ms = float(metrics.get("inference_ms", 0.0) or 0.0)
            dec_ms = float(metrics.get("decode_result_ms", 0.0) or 0.0)
            total_ms = prep_ms + inf_ms + dec_ms

            with self._metrics_lock:
                # Update accumulators
                self._total_batches += 1
                self._total_images_processed += int(batch_size)
                self._total_preprocessing_ms += prep_ms
                self._total_inference_ms += inf_ms
                self._total_decode_result_ms += dec_ms
                self._total_processing_ms += total_ms

                # Compute per-image means
                n_img = self._total_images_processed
                if n_img > 0:
                    self.performance_metrics["preprocessing_mean_ms"] = self._total_preprocessing_ms / n_img
                    self.performance_metrics["inference_mean_ms"] = self._total_inference_ms / n_img
                    self.performance_metrics["decode_result_mean_ms"] = self._total_decode_result_ms / n_img
                    self.performance_metrics["total_processing_mean_ms"] = self._total_processing_ms / n_img

                    total_sec = self._total_processing_ms / 1000.0
                    self.performance_metrics["throughput_images_per_sec"] = (n_img / total_sec) if total_sec > 0 else 0.0
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Failed to apply metrics: {e}")



    @staticmethod
    def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)


    def _decode_multiclass(self, logits_1d: np.ndarray, names: List[str]):
        # logits_1d: (C,)
        probs = self._softmax_np(logits_1d[None, :], axis=1)[0]
        idx = int(np.argmax(probs))
        label = names[idx] if 0 <= idx < len(names) else str(idx)
        return {"index": idx, "label": label, "prob": float(probs[idx]), "probs": probs.tolist()}

    @staticmethod
    def _decode_multilabel_or_singlelogit(logits_1d: np.ndarray, names: List[str], thr: float = 0.5):
        # sigmoid per class (or single logit)
        probs = 1.0 / (1.0 + np.exp(-logits_1d))
        preds = (probs >= thr).astype(int)
        return {"probs": probs.tolist(), "preds": preds.tolist(), "labels": names}


    def get_performance_metrics(self)->Dict[str,float]:
        # Return a snapshot to avoid external mutation and reduce race impact
        with self._metrics_lock:
            return dict(self.performance_metrics)

    def infer(
            self,
            id_img_list: List[Tuple[str, np.ndarray]],
            multilabel_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:

        # Warmup. Doesn't count towards performance metrics.
        # The first few batches are usually slower than later ones
        # due to initialization overhead.
        if self.warm_left > 0:
            self.warm_left -= 1
        
        metrics={}
        # Preprocess
        t0 = time.perf_counter()

        ids, batch = self.preprocessors.preprocess_list_id_bgr_threads(id_img_list)
        if len(ids) == 0:
            return []
        
        t1 = time.perf_counter()
        metrics["preprocessing_ms"] = (t1 - t0) * 1000.0

        # Inference
        output = self.session.session.run(None, {self.session.input_name: batch})
        t2 = time.perf_counter()
        metrics["inference_ms"] = (t2 - t1) * 1000.0

        # Map outputs to heads
        logits_per_head = {name: np.array(o) for name, o in zip(self.session.head_order, output)}

        # Decode heads
        results = []
        for i, _id in enumerate(ids):
            sample_heads: Dict[str, Any] = {
                "id": _id,
            }
            for head in self.session.head_order:
                logits_i = logits_per_head[head][i]
                names = self.session.class_names_map.get(head)
                htype=self.session.head_types.get(head)

                # Normalize to 1D logits for decoding
                if logits_i.ndim > 1:
                    logits_i = logits_i.reshape(-1)

                if htype in ("multilabel",) or logits_i.shape[0] == 1:
                    sample_heads[head] = self._decode_multilabel_or_singlelogit(logits_i, names, thr=multilabel_threshold)
                else:
                    sample_heads[head] = self._decode_multiclass(logits_i, names)
            results.append(sample_heads)
        metrics["decode_result_ms"] = (time.perf_counter() - t2) * 1000.0
        if self.warm_left == 0:
            self.update_metrics(len(ids),metrics)
        return results

