import cv2
import numpy as np
import os
import json
import time
from HailoInferClass import HailoInfer  # Replace accordingly


def preprocess_image_uint8(image_path, input_shape):
    # input_shape = (height, width, channels)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[0]))

    # No normalization, no mean/std
    # Convert directly to uint8
    img = img.astype(np.uint8)

    return img




def main():
    # --- Configuration ---
    hef_path = "/home/sir/Kevin_Walter/hailo/mobilenetv3_gtsrb_static.hef"
    images_dir = "/home/sir/Kevin_Walter/hailo/testing_accuracy_data"
    NUM_IMAGES = 4  # You can change to 1–16 depending on your batch size

    # --- Initialize the model ---
    hailo_infer = HailoInfer(
        hef_path=hef_path,
        batch_size=1,
        input_type="UINT8",
        output_type="UINT8"
    )

    # --- Get input shape ---
    input_shape = hailo_infer.get_input_shape()
    print("Model input shape:", input_shape)

    # --- Collect and preprocess images ---
    image_files = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        print("No images found in:", images_dir)
        return

    num_to_run = min(NUM_IMAGES, len(image_files))
    print(f"\nRunning synchronous inference on {num_to_run} images from {images_dir}")

    frames = [preprocess_image_uint8(p, input_shape) for p in image_files[:num_to_run]]

    # --- Run synchronous inference ---
    start_time = time.time()

    # Pass a batch (or list of 1) to the synchronous runner
    outputs = hailo_infer.run_sync(frames)

    total_time = time.time() - start_time
    print(f"\nInference completed in {total_time:.2f} seconds.")

    # --- Process and display results ---
    for i, result in enumerate(outputs):
        image_name = os.path.basename(image_files[i])
        top_label = result["top1_label"]
        confidence = result["top1_score"] * 100
        print(f"\n{image_name}")
        print(f"Top prediction: {top_label} ({confidence:.2f}%)")
        print("Top-5:")
        for rank, (cls_idx, prob) in enumerate(result["top5"], start=1):
            print(f"  {rank}. Class {cls_idx:<3d} — {prob*100:.2f}%")

    # --- Cleanup ---
    hailo_infer.close()
    print("\nDevice released successfully.")


if __name__ == "__main__":
    main()
