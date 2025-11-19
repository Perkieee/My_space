from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Config:
    imx_model_path: str = "/path_to.rpk"
    priors_path: str = "/path_to_ssd_priors.npy"
    preview_size: Tuple[int, int] = (640, 360)
    score_thresh: float = 0.6
    nms_iou_thresh: float = 0.45
    variances: Tuple[float, float] = (0.1, 0.2)
    class_name: str = "banana"

    hef_path: str = "/path_to.hef"
    hailo_batch_size: int = 16
    hailo_num_workers: int = 4
    hailo_input_shape: Tuple[int, int, int] = (224, 224, 3)
    hailo_throughput_mode: str = "end_to_end"

    crop_thread_workers: int = 4
    batch_submit_thread_count: int = 1
    batch_max_size: int = 16
    queue_maxsize: int = 8
    run_time: float = 15.0

    show_preview: bool = True
    verbose: bool = False

    test_mode: bool = False
    test_frames: int = 200
    json_output: Optional[str] = None
