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

    def submit_batch(self, frames: List[np.ndarray], metas: List[Dict], callback_fn):
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
                    self.model.run_async(frames, callback_fn, metadata=batch_meta)
                except TypeError:
                    self.model.run_async(frames, callback_fn)
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
                    callback_fn(bindings_list, batch_meta)

                import threading
                threading.Thread(target=_mock_job, daemon=True).start()

        except Exception as e:
            self.logger.error("Hailo run_async failed: %s", e)

        t1 = time.perf_counter()
        host_enqueue_ms = (t1 - t0) * 1000.0
        self.hailo_host_enqueue_latencies.append(host_enqueue_ms)

        return host_enqueue_ms
