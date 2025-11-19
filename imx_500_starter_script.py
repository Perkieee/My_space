# Author: Kevin Walter
# imx_500_starter_script.py
# Display for IMX500 SSDLite320 -> 3234 priors
# Key notes:
#  - capture_request() to get image + metadata from the same camera request
#  - early thresholding, vectorized ops, compact preview resolution
#  - simple timings shown on-screen
# NOTE: This is a starter script to help verify the quantized model is running on the imx500.
#       The production file utilizes asyncio for parellization and will be implemented.

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500

# ------------- user config -------------
MODEL_PATH = "/path_to.rpk file"
PRIORS_PATH = "/path_to_ssd_priors.npy file" # I generated anchors prior for the postprocessing. feel free to switch to your own style
PREVIEW_SIZE = (640, 360)   # smaller = less drawing overhead
SCORE_THRESH = 0.6
NMS_IOU_THRESH = 0.45
VARIANCES = [0.1, 0.2]
CLASS_NAME = "banana" # switch class names according to your training
RUN_TIME = 15.0  # seconds to run
# ---------------------------------------

def load_priors(path):
    p = np.load(path)
    print(f"Loaded priors ({len(p)}) from {path}")
    return p

def decode_boxes(locs, priors, variances):
    # locs: (N,4) [dx,dy,dw,dh]
    cx = priors[:, 0]; cy = priors[:, 1]; pw = priors[:, 2]; ph = priors[:, 3]
    dx = locs[:, 0]; dy = locs[:, 1]; dw = locs[:, 2]; dh = locs[:, 3]
    decoded_cx = dx * variances[0] * pw + cx
    decoded_cy = dy * variances[0] * ph + cy
    decoded_w = np.exp(dw * variances[1]) * pw
    decoded_h = np.exp(dh * variances[1]) * ph
    xmin = decoded_cx - decoded_w / 2.0
    ymin = decoded_cy - decoded_h / 2.0
    xmax = decoded_cx + decoded_w / 2.0
    ymax = decoded_cy + decoded_h / 2.0
    boxes = np.stack([xmin, ymin, xmax, ymax], axis=1)
    return np.clip(boxes, 0.0, 1.0)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def nms(boxes, scores, iou_threshold=0.45):
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)

def draw_overlay(frame, detections, fps, inf_ms, post_ms):
    # detections: list of tuples (x1,y1,x2,y2,score)
    H, W = frame.shape[:2]
    # draw boxes
    for (xmin, ymin, xmax, ymax, score) in detections:
        x1 = int(xmin * W); y1 = int(ymin * H)
        x2 = int(xmax * W); y2 = int(ymax * H)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{CLASS_NAME} {score:.2f}"
        cv2.putText(frame, label, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # draw debug text
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Inf: {inf_ms:.1f} ms", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, f"Post: {post_ms:.1f} ms", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

def main():
    priors = load_priors(PRIORS_PATH)
    expected = 3234
    if len(priors) != expected:
        print(f"WARNING priors len {len(priors)} != expected {expected} -> decoding may be wrong")

    imx500 = IMX500(MODEL_PATH)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": PREVIEW_SIZE, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print("Camera + IMX500 model started. Warming up AWB/AE for 2s...")
    time.sleep(2.0)

    win_name = "IMX500 Starter Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    start_time = time.time()
    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()

    while time.time() - start_time < RUN_TIME:
        # Capture a single request (frame + metadata) so boxes match frame exactly
        req = picam2.capture_request()   # -> a Request object
        # Obtain frame from the 'main' stream
        try:
            frame = req.make_array("main")  # same frame as metadata
        except Exception:
            # fallback to capture_array if API differs
            frame = picam2.capture_array()
        metadata = req.get_metadata()

        t_inf0 = time.time()
        outputs = imx500.get_outputs(metadata)  # list-like: [locs, confs]
        t_inf1 = time.time()

        detections = []
        if outputs and len(outputs) >= 2:
            locs = np.asarray(outputs[0], dtype=np.float32)
            confs = np.asarray(outputs[1], dtype=np.float32)

            # Quick sanity check
            if locs.shape[0] == priors.shape[0]:
                t_post0 = time.time()
                # Decode boxes (normalized)
                boxes = decode_boxes(locs, priors, VARIANCES)  # (N,4)
                # softmax -> class probs
                probs = softmax(confs, axis=1)
                # scores for banana (index 1)
                scores = probs[:, 1]
                # early filter by score
                keep_mask = scores > SCORE_THRESH
                if np.any(keep_mask):
                    boxes_f = boxes[keep_mask]
                    scores_f = scores[keep_mask]
                    # run NMS on filtered set
                    keep_idxs = nms(boxes_f, scores_f, iou_threshold=NMS_IOU_THRESH)
                    for idx in keep_idxs:
                        b = boxes_f[idx]
                        s = scores_f[idx]
                        detections.append((b[0], b[1], b[2], b[3], float(s)))
                t_post1 = time.time()
                post_ms = (t_post1 - t_post0) * 1000.0
            else:
                post_ms = 0.0
        else:
            post_ms = 0.0

        inf_ms = (t_inf1 - t_inf0) * 1000.0

        # Release the request so libcamera can reuse buffers
        try:
            req.release()
        except Exception:
            pass

        # Update FPS
        frame_count += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now

        # Draw overlays on frame (in-place)
        draw_overlay(frame, detections, fps, inf_ms, post_ms)

        cv2.imshow(win_name, frame)
        # press ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
