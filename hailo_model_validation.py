#!/usr/bin/env python3

"""

Refined Hailo-8 validation/inference pipeline



Features:

- Configuration via dataclass (InferenceConfig)

- Threaded CPU preprocessing with ThreadPoolExecutor

- Correct per-batch accuracy and overall accuracy (thread-safe)

- Throughput measurement modes: device, host, end_to_end

- Verbose logging toggle to avoid throughput degradation

- Preserves HailoInfer API usage: run_async(...), last_infer_job

"""



from __future__ import annotations



import argparse

import json

import logging

import os

import threading

import time

from concurrent.futures import ThreadPoolExecutor, Future

from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple



import cv2

import numpy as np

import pandas as pd



# Import existing wrapper - must be available in PYTHONPATH

from HailoInferClass import HailoInfer





# -------------------------

# Configuration dataclass

# -------------------------

@dataclass

class InferenceConfig:

    hef: str

    images_dir: str

    class_names: str

    csv_labels: Optional[str] = None

    output_dir: Optional[str] = None

    batch_size: int = 24

    num_workers: int = 4

    num_images: Optional[int] = None

    log_level: str = "INFO"

    verbose: bool = False

    throughput_mode: str = "end_to_end"  # options: device | host | end_to_end

    device_id: int = 0  # retained for future device selection compatibility





# -------------------------

# Utility functions

# -------------------------

def preprocess_image_uint8(image_path, input_shape):

    # input_shape = (height, width, channels)

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (input_shape[1], input_shape[0]))



    # No normalization, no mean/std

    # Convert directly to uint8

    img = img.astype(np.uint8)



    return img





def load_labels_from_csv(csv_path: str) -> Dict[str, int]:

    """

    Load CSV mapping of filenames to class IDs. Expects columns 'ConvertedFilename' and 'ClassId'.

    Returns mapping {ConvertedFilename: ClassId}

    """

    df = pd.read_csv(csv_path)

    if "ConvertedFilename" not in df.columns or "ClassId" not in df.columns:

        raise ValueError("CSV must include 'ConvertedFilename' and 'ClassId' columns.")

    return {row["ConvertedFilename"]: int(row["ClassId"]) for _, row in df.iterrows()}





def load_class_names(json_path: str) -> Dict[int, str]:

    """

    Load class names JSON in the format:

      {"0": [0, "Speed limit (20km/h)"], "1": [1, "Speed limit (30km/h)"], ...}

    Returns mapping {0: "Speed limit (20km/h)", ...}

    """

    with open(json_path, "r") as f:

        labels_json = json.load(f)

    return {int(k): v[1] for k, v in labels_json.items()}





# -------------------------

# HailoApp class

# -------------------------

class HailoApp:

    """

    Encapsulates the Hailo inference pipeline with threaded preprocessing and

    accurate metrics (batch & overall accuracy) and throughput measurement.

    """



    def __init__(self, config: InferenceConfig) -> None:

        self.config = config



        # Model and input info (populated in load_model)

        self.model: Optional[HailoInfer] = None

        self.input_shape: Optional[Tuple[int, int, int]] = None



        # Load class names immediately

        self.class_names = load_class_names(self.config.class_names)



        # Image files (sorted deterministic)

        if not os.path.isdir(self.config.images_dir):

            raise FileNotFoundError(f"Images directory not found: {self.config.images_dir}")



        all_files = sorted(os.listdir(self.config.images_dir))

        self.image_names = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        self.image_files = [os.path.join(self.config.images_dir, f) for f in self.image_names]



        # Optional CSV true labels mapping

        self.true_labels_map: Dict[str, int] = {}

        if self.config.csv_labels:

            if os.path.exists(self.config.csv_labels):

                self.true_labels_map = load_labels_from_csv(self.config.csv_labels)

            else:

                logging.warning("CSV labels path provided but not found: %s. Continuing without CSV labels.", self.config.csv_labels)

                self.config.csv_labels = None



        # Prepare filenames_for_hailo mapping to match original expected structure:

        # {index: [image_name, true_class_or_None]}

        self.filenames_for_hailo: Dict[int, List] = {}

        for i, image_name in enumerate(self.image_names):

            true_cls = self.true_labels_map.get(image_name)

            self.filenames_for_hailo[i] = [image_name, true_cls if true_cls is not None else None]



        # Thread-safe accumulators for accuracy

        self._lock = threading.Lock()

        self.all_truths: List[str] = []

        self.all_predictions: List[str] = []

        self.correct_total: int = 0

        self.total_samples: int = 0

        self.batch_accuracies: List[float] = []  # per-batch accuracy values



        # Timing fields for throughput metrics

        self._end_to_end_start: Optional[float] = None

        self._end_to_end_end: Optional[float] = None

        self._host_start: Optional[float] = None

        self._host_end: Optional[float] = None

        self._device_start: Optional[float] = None

        self._device_end: Optional[float] = None



    # ---------------------------

    # Model lifecycle

    # ---------------------------

    def load_model(self) -> None:

        """Instantiate HailoInfer and query input shape."""

        self.model = HailoInfer(

            hef_path=self.config.hef,

            batch_size=self.config.batch_size,

            filenames=self.filenames_for_hailo,

            input_type="UINT8",

            output_type="UINT8",

        )

        self.input_shape = self.model.get_input_shape()

        logging.info("Model input shape: %s", str(self.input_shape))



    def close(self) -> None:

        """Close/release Hailo resources."""

        if self.model:

            try:

                self.model.close()

            except Exception as e:

                logging.warning("Exception while closing model: %s", e)

            finally:

                logging.info("Hailo device released.")



    # ---------------------------

    # Batching / Preprocessing

    # ---------------------------

    def _image_batches(self) -> List[List[str]]:

        """Create deterministic batches of image file paths."""

        if not self.image_files:

            return []

        num_to_run = self.config.num_images if self.config.num_images is not None else len(self.image_files)

        num_to_run = min(num_to_run, len(self.image_files))

        files = self.image_files[:num_to_run]

        batches: List[List[str]] = []

        for i in range(0, len(files), self.config.batch_size):

            batches.append(files[i : i + self.config.batch_size])

        return batches



    def prepare_batch(self, batch_files: List[str]) -> List[np.ndarray]:

        """Preprocess a batch of images (returns list of uint8 frames)."""

        assert self.input_shape is not None, "Input shape must be available (call load_model first)."

        frames: List[np.ndarray] = []

        for p in batch_files:

            try:

                frame = preprocess_image_uint8(p, self.input_shape)

            except Exception as e:

                logging.warning("Failed to preprocess %s: %s", p, e)

                continue

            frames.append(frame)

        return frames



    # ---------------------------

    # Inference callback

    # ---------------------------





    def _extract_true_from_meta(self, meta: dict) -> tuple[Optional[str], Optional[str]]:

        """

        Robustly extract (true_label_str, filename) from Hailo metadata.



        Works across both single- and multi-batch runs.



        Returns:

            (true_label, filename) where:

              - true_label: readable string like "Speed limit (30km/h)"

              - filename: original image filename

        """

        # Prefer index-based lookup ï¿½ most reliable for multi-batch async inference

        idx = meta.get("index")

        if idx is not None:

            try:

                idx = int(idx)

                if idx in self.filenames_for_hailo:

                    filename, true_cls = self.filenames_for_hailo[idx]

                    if true_cls is not None:

                        true_label = self.class_names.get(true_cls, f"Class {true_cls}")

                        return true_label,filename

            except Exception:

                pass



    def inference_callback(self, infer_results=None, bindings_list=None, validate: bool = True, **kwargs) -> None:

        """

        Callback invoked by HailoInfer.run_async. This postprocessing contains:

        - softmax, top-5, mapping to class names

        - computes per-batch accuracy and appends to global accumulators in a thread-safe way

        """

        if bindings_list is None:

            logging.warning("No bindings_list received in callback.")

            return



        batch_truths: List[str] = []

        batch_preds: List[str] = []



        for info in bindings_list:

            binding = info.get("binding")

            meta = info.get("metadata", {})

            # Get output buffer(s)

            try:

                output_buffers = binding.output().get_buffer()

            except Exception as e:

                logging.warning("Failed to get output buffer: %s", e)

                continue



            if isinstance(output_buffers, dict):

                first_output = list(output_buffers.values())[0]

            else:

                first_output = output_buffers



            output_flat = first_output.flatten().astype(np.float32)

            exp_scores = np.exp(output_flat - np.max(output_flat))

            probs = exp_scores / np.sum(exp_scores)



            top5_idx = np.argsort(probs)[-5:][::-1]

            top_pred_idx = int(top5_idx[0])

            confidence_pct = float(probs[top_pred_idx] * 100.0)



            pred_label = self.class_names.get(top_pred_idx, f"Class {top_pred_idx}")

            # Robust label + filename lookup

            true_label, filename_from_meta = self._extract_true_from_meta(meta)



            # Optional debug print (only if verbose enabled)

            if self.config.verbose:

                logging.debug(f"[DEBUG] index={meta.get('index')} filename={filename_from_meta} "

                              f"true_label={true_label} pred_label={pred_label}")



            if validate and true_label is not None:

                batch_truths.append(true_label)

                batch_preds.append(pred_label)



            # Verbose per-image logging (avoid by default)

            if self.config.verbose:

                logging.debug("Image: %s Pred: %s (%.2f%%)", filename_from_meta, pred_label, confidence_pct)



        # Compute batch accuracy and update global accumulators safely

        if validate and batch_truths:

            truths_arr = np.array(batch_truths)

            preds_arr = np.array(batch_preds)

            batch_accuracy = float(np.mean(preds_arr == truths_arr))

            with self._lock:

                # update overall counters

                self.correct_total += int(np.sum(preds_arr == truths_arr))

                self.total_samples += len(truths_arr)

                self.all_truths.extend(batch_truths)

                self.all_predictions.extend(batch_preds)

                self.batch_accuracies.append(batch_accuracy)



            logging.info("Batch processed: size=%d batch_accuracy=%.2f%%", len(truths_arr), batch_accuracy * 100.0)

        else:

            # If no validation labels present in this batch, still log size if verbose

            if self.config.verbose:

                logging.debug("Batch processed with no validation labels or none matched.")



    # ---------------------------

    # Run pipeline

    # ---------------------------

    def run(self) -> None:

        """Orchestrate threaded preprocessing and asynchronous inference, measuring throughput per config."""

        if not self.image_files:

            logging.error("No images found in: %s", self.config.images_dir)

            return



        if self.model is None or self.input_shape is None:

            self.load_model()



        batches = self._image_batches()

        if not batches:

            logging.error("No batches created from images.")

            return



        # Prepare output directory

        if self.config.output_dir:

            os.makedirs(self.config.output_dir, exist_ok=True)



        # Start end-to-end timer

        if self.config.throughput_mode == "end_to_end":

            self._end_to_end_start = time.perf_counter()



        # ThreadPool for CPU preprocessing

        with ThreadPoolExecutor(max_workers=max(1, self.config.num_workers)) as executor:

            preprocess_futures: List[Tuple[int, Future]] = []

            next_batch_idx = 0

            first_run_async_called = False



            # Pre-submit first preprocess job

            if next_batch_idx < len(batches):

                fut = executor.submit(self.prepare_batch, batches[next_batch_idx])

                preprocess_futures.append((next_batch_idx, fut))

                next_batch_idx += 1



            # Host timing begins just before we start calling run_async (surrounding run_async calls)

            self._host_start = time.perf_counter()



            # Loop until all preprocess jobs complete and are submitted to device

            while preprocess_futures:

                idx, future = preprocess_futures.pop(0)

                frames = future.result()  # may raise; it's okay to let this surface to caller

                if not frames:

                    # Nothing to run for this batch (maybe preprocessing failed for all images)

                    if self.config.verbose:

                        logging.debug("No frames generated for batch %d", idx)

                else:

                    # For device throughput timing: record when we enqueue the first device job

                    if not first_run_async_called:

                        self._device_start = time.perf_counter()

                        first_run_async_called = True



                    # Submit to Hailo (preserve original API: run_async(frames, callback))

                    assert self.model is not None

                    try:

                        # Note: run_async is expected not to block until device done (async)

                        self.model.run_async(frames, self.inference_callback)

                    except Exception as e:

                        logging.error("Error when calling model.run_async: %s", e)

                        # continue to next batch but don't crash entire loop



                # Immediately submit preprocessing for next batch (if any)

                if next_batch_idx < len(batches):

                    fut = executor.submit(self.prepare_batch, batches[next_batch_idx])

                    preprocess_futures.append((next_batch_idx, fut))

                    next_batch_idx += 1



            # All run_async calls have been submitted at this point -> host end time

            self._host_end = time.perf_counter()



            # Wait for last device job to finish (preserve original behavior using last_infer_job)

            if self.model and getattr(self.model, "last_infer_job", None):

                last_job = self.model.last_infer_job

                try:

                    if hasattr(last_job, "wait"):

                        # Wait for device to complete; preserve a timeout behavior similar to original

                        last_job.wait(1000)

                    else:

                        # fallback small sleep to allow device finish

                        time.sleep(0.1)

                except Exception as e:

                    logging.warning("Exception while waiting for last_infer_job: %s", e)



            # Device end time is after last job completion

            if first_run_async_called:

                self._device_end = time.perf_counter()



        # End-to-end timer end

        if self.config.throughput_mode == "end_to_end":

            self._end_to_end_end = time.perf_counter()



        # Close resources

        self.close()



        # Final metrics reporting

        self._report_metrics()



    # ---------------------------

    # Metrics & reporting

    # ---------------------------

    def _report_metrics(self) -> None:

        """Compute and log/print throughput and accuracies as requested."""

        num_images = self.config.num_images if self.config.num_images is not None else len(self.image_files)



        # Overall accuracy

        print(f'overall correct total: {self.correct_total}')

        print(f'total images processed: {self.total_samples}')

        print(f'Overall accuracy: {(self.correct_total/self.total_samples)*100} %')

        overall_acc = (self.correct_total / self.total_samples) if self.total_samples > 0 else None

        avg_batch_acc = float(np.mean(self.batch_accuracies)) if self.batch_accuracies else None



        # Compute throughput values (images / second) for all three metrics if possible

        device_throughput = None

        host_throughput = None

        end_to_end_throughput = None



        # Device throughput: from first enqueue to last job completion

        if self._device_start is not None and self._device_end is not None:

            device_elapsed = max(1e-9, self._device_end - self._device_start)

            device_throughput = num_images / device_elapsed



        # Host throughput: surrounding run_async calls (enqueueing)

        if self._host_start is not None and self._host_end is not None:

            logging.info("Total images processed (labeled used for accuracy): %d", self.total_samples)



        # Choose which throughput to highlight based on config

        selected_mode = self.config.throughput_mode.lower()

        if selected_mode not in {"device", "host", "end_to_end"}:

            logging.warning("Unknown throughput_mode '%s', defaulting to end_to_end", self.config.throughput_mode)

            selected_mode = "end_to_end"



        # Also print all measured throughputs at debug/info level so user can compare

        logging.debug("All throughputs: device=%s host=%s end_to_end=%s",

                      f"{device_throughput:.2f}" if device_throughput else "N/A",

                      f"{host_throughput:.2f}" if host_throughput else "N/A",

                      f"{end_to_end_throughput:.2f}" if end_to_end_throughput else "N/A")



        # Accuracies

        # if avg_batch_acc is not None:

        #     logging.info("Average Batch Accuracy: %.2f%% (averaged across batches with validation)", avg_batch_acc * 100.0)

        # else:

        #     logging.info("Average Batch Accuracy: N/A (no labeled batches processed)")



        if overall_acc is not None:

            logging.info("Final Overall Accuracy: %.2f%% (%d/%d)", overall_acc * 100.0, self.correct_total, self.total_samples)

        else:

            logging.info("Final Overall Accuracy: N/A (no labeled samples processed)")



        # If verbose, show per-batch accuracies

        if self.config.verbose and self.batch_accuracies:

            for i, a in enumerate(self.batch_accuracies):

                logging.debug("Batch %d accuracy: %.2f%%", i, a * 100.0)





# -------------------------

# CLI and entrypoint

# -------------------------

def parse_args() -> InferenceConfig:

    p = argparse.ArgumentParser(description="Refined Hailo-8 inference pipeline")

    p.add_argument("--hef", required=True, help="Path to compiled Hailo HEF model.")

    p.add_argument("--images-dir", required=True, help="Directory with input images (.jpg/.png/.jpeg).")

    p.add_argument("--class-names", required=True, help="Path to class names JSON (format: {\"0\":[0,\"label\"],...}).")

    p.add_argument("--csv-labels", required=False, default=None, help="Optional CSV with true labels (ConvertedFilename,ClassId).")

    p.add_argument("--output-dir", required=False, default=None, help="Optional output directory to save results.")

    p.add_argument("--batch-size", required=False, type=int, default=24, help="Batch size for inference (default: 24).")

    p.add_argument("--num-workers", required=False, type=int, default=4, help="Number of CPU threads for preprocessing (default: 4).")

    p.add_argument("--num-images", required=False, type=int, default=None, help="Number of images to run (default: all).")

    p.add_argument("--log-level", required=False, default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR).")

    p.add_argument("--verbose", action="store_true", help="Enable verbose per-image logging (can reduce throughput).")

    p.add_argument("--throughput-mode", required=False, default="end_to_end", choices=["device", "host", "end_to_end"],

                   help="Which throughput metric to report (device|host|end_to_end).")

    args = p.parse_args()



    config = InferenceConfig(

        hef=args.hef,

        images_dir=args.images_dir,

        class_names=args.class_names,

        csv_labels=args.csv_labels,

        output_dir=args.output_dir,

        batch_size=args.batch_size,

        num_workers=args.num_workers,

        num_images=args.num_images,

        log_level=args.log_level,

        verbose=args.verbose,

        throughput_mode=args.throughput_mode,

    )

    return config





def main() -> None:

    config = parse_args()

    numeric_level = getattr(logging, config.log_level.upper(), logging.INFO)

    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")



    # Basic validations

    if not os.path.exists(config.hef):

        logging.error("HEF model not found: %s", config.hef)

        return



    try:

        app = HailoApp(config)

    except Exception as e:

        logging.error("Failed to initialize HailoApp: %s", e)

        return



    try:

        app.run()

    finally:

        # Ensure resources released

        app.close()





if __name__ == "__main__":

    main()

