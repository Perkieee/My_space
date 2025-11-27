#!/usr/bin/env python3

# Author: Kevin Walter

"""

Hailo Synchronous Inference Runner

----------------------------------

Run classification inference on a Hailo-8 device using the synchronous API.

Supports accuracy computation, timing metrics, and optional verbose outputs.

"""



import os

import cv2

import json

import time

import argparse

import numpy as np

import pandas as pd

from typing import Dict, List, Tuple, Any



from HailoInferClass import HailoInfer  # your existing class





# -----------------------------------------------------------

# Utility functions

# -----------------------------------------------------------



def preprocess_image_uint8(image_path: str, input_shape: Tuple[int, int, int]) -> np.ndarray:

    """Resize and convert image to uint8 without normalization."""

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (input_shape[1], input_shape[0]))

    return img.astype(np.uint8)





def load_labels_from_csv(csv_path: str) -> Dict[str, int]:

    """Load image filename â†’ class_id mapping from CSV."""

    df = pd.read_csv(csv_path)

    return {row["ConvertedFilename"]: int(row["ClassId"]) for _, row in df.iterrows()}





def load_class_names(json_path: str) -> Dict[int, str]:

    """Load class index â†’ readable name mapping from JSON."""

    with open(json_path, "r") as f:

        labels_json = json.load(f)

    return {int(k): v[1] for k, v in labels_json.items()}





# -----------------------------------------------------------

# Core inference manager

# -----------------------------------------------------------



class HailoSyncSession:

    """Manage synchronous inference, metrics, and accuracy evaluation."""



    def __init__(self, hef_path: str, batch_size: int = 1,

                 input_type: str = "UINT8", output_type: str = "UINT8"):

        self.model = HailoInfer(

            hef_path=hef_path,

            batch_size=batch_size,

            input_type=input_type,

            output_type=output_type

        )

        self.input_shape = self.model.get_input_shape()



    def infer_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:

        """Run synchronous inference on a batch of preprocessed images."""

        return self.model.run_sync(images)



    def close(self):

        """Release Hailo device and resources."""

        self.model.close()





# -----------------------------------------------------------

# Accuracy and reporting helpers

# -----------------------------------------------------------



def compute_accuracy(predictions: np.ndarray, truths: np.ndarray,

                     verbose: bool = False) -> None:

    """Compute and print global and per-class accuracy."""

    accuracy = np.mean(predictions == truths)

    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")



    if verbose:

        unique_labels = sorted(set(truths))

        print("\nPer-class accuracy:")

        for label in unique_labels:

            mask = truths == label

            correct = np.sum(predictions[mask] == truths[mask])

            total = np.sum(mask)

            acc = (correct / total) * 100 if total > 0 else 0

            print(f"{label}: {acc:.2f}% ({correct}/{total})")





# -----------------------------------------------------------

# Main inference logic

# -----------------------------------------------------------



def run_inference(args):

    """Main entry for inference process."""

    # Load label data

    class_names = load_class_names(args.labels_json)

    true_labels_map = load_labels_from_csv(args.labels_csv)



    # Initialize model

    hailo_session = HailoSyncSession(args.hef_path, batch_size=args.batch_size)

    print("Model input shape:", hailo_session.input_shape)



    # Collect images

    image_files = [

        os.path.join(args.images_dir, f)

        for f in sorted(os.listdir(args.images_dir))

        if f.lower().endswith((".jpg", ".jpeg", ".png"))

    ]

    if not image_files:

        print("No images found in:", args.images_dir)

        return



    num_to_run = min(args.num_images, len(image_files))

    print(f"\nRunning synchronous inference on {num_to_run} images...")



    frames = [preprocess_image_uint8(p, hailo_session.input_shape)

              for p in image_files[:num_to_run]]



    start_time = time.time()

    outputs = hailo_session.infer_batch(frames)

    total_time = time.time() - start_time

    print(f"\nInference completed in {total_time:.2f} seconds.")



    # Evaluate results

    predictions, truths = [], []

    for i, result in enumerate(outputs):

        image_name = os.path.basename(image_files[i])

        pred_label_idx = result["top1_label"]

        pred_label = class_names[pred_label_idx]

        confidence = result["top1_score"] * 100

        true_label = class_names[true_labels_map[image_name]]



        predictions.append(pred_label)

        truths.append(true_label)



        print(f"\n {image_name}")

        print(f"True label: {true_label}")

        print(f"Predicted: {pred_label} ({confidence:.2f}%)")



        if args.verbose:

            print("Top-5 predictions:")

            for rank, (cls_idx, prob) in enumerate(result["top5"], start=1):

                print(f"  {rank}. {class_names[cls_idx]} â€” {prob*100:.2f}%")



    # Compute accuracy

    compute_accuracy(np.array(predictions), np.array(truths), verbose=args.verbose)



    # Cleanup

    hailo_session.close()

    print("\nDevice released successfully.")





# -----------------------------------------------------------

# CLI entry point

# -----------------------------------------------------------



def parse_args():

    parser = argparse.ArgumentParser(

        description="Run synchronous inference on Hailo-8 device."

    )

    parser.add_argument("--hef-path", required=True, help="Path to compiled HEF file.")

    parser.add_argument("--images-dir", required=True, help="Directory containing input images.")

    parser.add_argument("--labels-json", required=True, help="Path to class names JSON.")

    parser.add_argument("--labels-csv", required=True, help="CSV with true labels (filename -> class id).")

    parser.add_argument("--num-images", type=int, default=10, help="Number of images to run inference on.")

    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synchronous inference.")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (top-5, per-class accuracy).")

    return parser.parse_args()





def main():

    args = parse_args()

    run_inference(args)





if __name__ == "__main__":

    main()

