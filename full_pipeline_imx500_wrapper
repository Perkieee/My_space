import time
import logging
import numpy as np
from typing import Tuple, List

try:
    from picamera2.devices import IMX500
except Exception:
    IMX500 = None


class IMX500Wrapper:
    def __init__(self, model_path, priors_path, variances, score_thresh, nms_thresh):
        self.model_path = model_path
        self.priors = self.load_priors(priors_path)
        self.variances = variances
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        if IMX500 is not None:
            try:
                self.model = IMX500(model_path)
            except Exception as e:
                logging.warning("Failed to instantiate IMX500: %s", e)
                self.model = None
        else:
            self.model = None

    def load_priors(self, path: str):
        priors = np.load(path)
        logging.info("Loaded priors (%d) from %s", len(priors), path)
        return priors

    @staticmethod
    def decode_boxes(locs, priors, variances):
        cx = priors[:, 0]; cy = priors[:, 1]
        pw = priors[:, 2]; ph = priors[:, 3]
        dx = locs[:, 0]; dy = locs[:, 1]
        dw = locs[:, 2]; dh = locs[:, 3]

        decoded_cx = dx * variances[0] * pw + cx
        decoded_cy = dy * variances[0] * ph + cy
        decoded_w = np.exp(dw * variances[1]) * pw
        decoded_h = np.exp(dh * variances[1]) * ph

        xmin = decoded_cx - decoded_w / 2
        ymin = decoded_cy - decoded_h / 2
        xmax = decoded_cx + decoded_w / 2
        ymax = decoded_cy + decoded_h / 2

        return np.clip(np.stack([xmin, ymin, xmax, ymax], axis=1), 0.0, 1.0)

    @staticmethod
    def softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    @staticmethod
    def nms(boxes, scores, iou_threshold=0.45):
        if len(boxes) == 0:
            return np.array([], dtype=int)

        x1 = boxes[:, 0]; y1 = boxes[:, 1]
        x2 = boxes[:, 2]; y2 = boxes[:, 3]
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
            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return np.array(keep, dtype=int)

    def get_detections_from_metadata(self, metadata):
        if self.model is None:
            return [], 0.0, 0.0

        t0 = time.perf_counter()
        outputs = self.model.get_outputs(metadata)
        t1 = time.perf_counter()
        inf_ms = (t1 - t0) * 1000

        detections = []
        post_ms = 0.0

        if outputs and len(outputs) >= 2:
            locs = np.asarray(outputs[0], np.float32)
            confs = np.asarray(outputs[1], np.float32)

            if locs.shape[0] == self.priors.shape[0]:
                t2 = time.perf_counter()
                boxes = self.decode_boxes(locs, self.priors, self.variances)
                probs = self.softmax(confs, axis=1)
                scores = probs[:, 1]

                keep_mask = scores > self.score_thresh
                if np.any(keep_mask):
                    boxes_f = boxes[keep_mask]
                    scores_f = scores[keep_mask]
                    keep = self.nms(boxes_f, scores_f, self.nms_thresh)

                    for idx in keep:
                        b = boxes_f[idx]
                        detections.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(scores_f[idx])))

                t3 = time.perf_counter()
                post_ms = (t3 - t2) * 1000

        return detections, inf_ms, post_ms
