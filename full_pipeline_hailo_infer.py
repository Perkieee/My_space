import time
import uuid
import logging
from typing import List, Dict
import numpy as np
from collections import deque

try:
    from HailoInferClass import HailoInfer
except Exception:
    HailoInfer = None


class HailoWrapper:
    def __init__(self, hef_path, batch_size, input_shape, logger):
        self.hef_path = hef_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.logger = logger

        if HailoInfer is None:
            logger.warning("HailoInfer not importable; running in mock mode")
            self.model = None
        else:
            try:
                self.model = HailoInfer(
                    hef_path=hef_path,
                    batch_size=batch_size,
                    filenames={},
                    input_type="UINT8",
                    output_type="UINT8",
                )
            except Exception as e:
                logger.warning("Failed to create HailoInfer: %s", e)
                self.model = None

        self.mock_mode = self.model is None

        self.lock = logging.Lock()
        self.total_batches_submitted = 0
        self.total_samples_submitted = 0

        self.job_enqueue_times = {}
        self.hailo_device_latencies = deque(maxlen=1000)
        self.hailo_host_enqueue_latencies = deque(maxlen=1000)
        self.hailo_end_to_end_latencies = deque(maxlen=1000)

    def close(self):
        if self.model:
            try:
                self.model.close()
            except Exception as e:
                self.logger.warning("Error closing Hailo: %s", e)
    def inference_callback(self, infer_results=None, bindings_list=None, **kwargs):
        """
        Pure inference callback for Hailo run_async().
        - Reads output buffers
        - Applies softmax
        - Extracts top-1 + top-5 predictions
        - Logs prediction summary
        """

        if bindings_list is None:
            self.logger.warning("No bindings_list received in callback.")
            return

        for info in bindings_list:
            binding = info.get("binding")

            try:
                output_buffers = binding.output().get_buffer()
            except Exception as e:
                self.logger.error(f"Failed to get output buffer: {e}")
                continue

            # Handle dict or array output
            if isinstance(output_buffers, dict):
                out = list(output_buffers.values())[0]
            else:
                out = output_buffers

            # Convert to float32 flat vector
            logits = out.flatten().astype(np.float32)

            # Softmax
            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / np.sum(exp_scores)

            # Top-5
            top5_idx = probs.argsort()[-5:][::-1]
            top1_idx = int(top5_idx[0])
            top1_conf = float(probs[top1_idx] * 100.0)

            # Map class ID → label
            pred_label = self.class_names.get(top1_idx, f"class_{top1_idx}")

            # Log prediction
            self.logger.info(
                f"[HAILO] Pred: {pred_label} ({top1_conf:.2f}%) | Top5: {top5_idx.tolist()}"
            )


    def submit_batch(self, frames: List[np.ndarray], metas: List[Dict]):
        batch_id = str(uuid.uuid4())
        batch_meta = {
            "batch_id": batch_id,
            "samples": metas,
            "enqueue_time": time.perf_counter(),
        }

        t0 = time.perf_counter()

        try:
            if self.model:
                try:
                    self.model.run_async(frames, self.inference_callback, metadata=batch_meta)
                except TypeError:
                    self.model.run_async(frames, self.inference_callback)
            else:
                def _mock_job():
                    time.sleep(0.02 + 0.005 * len(frames))
                    bindings_list = [
                        {
                            "binding": {"output": lambda: np.random.rand(1000)},
                            "metadata": {"batch_id": batch_id, "sample_index": i},
                        }
                        for i in range(len(frames))
                    ]
                    self.inference_callback(bindings_list, batch_meta)

                import threading
                threading.Thread(target=_mock_job, daemon=True).start()

        except Exception as e:
            self.logger.error("Hailo run_async failed: %s", e)

        t1 = time.perf_counter()
        host_enqueue_ms = (t1 - t0) * 1000.0
        self.hailo_host_enqueue_latencies.append(host_enqueue_ms)

        return host_enqueue_ms
