
import json
from typing import List

import torch
from torch.utils.data import DataLoader

from lib.utils import (get_device, is_notebook)
from lib.inference.onnx_inference import ONNXInference
from lib.model import ModelHeads
from lib.trainer import Metrics

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm.std import tqdm


class ONNXEvaluation:

    def __init__(self,onnx_model_path:str,test_data_set:torch.utils.data.Dataset,heads_cfg:ModelHeads,batch_size:int=64):
        self.onnx_model_path = onnx_model_path
        self.test_data_set = test_data_set
        self.heads_cfg = heads_cfg
        self.batch_size = batch_size
        self.device = get_device()
        self.inference = ONNXInference(self.onnx_model_path)
        self.inference.initialize()
        self.metrics = Metrics()
        self.metrics.heads=heads_cfg.heads.keys()

    def build_inference_batch(self, batch_count, images, targets) -> List[tuple]:
        res = []
        img_index = (batch_count - 1) * self.batch_size
        for image, label in zip(images, targets):
            image_id = json.dumps({"id": str(img_index), "label": label})
            res.append((image_id, image))
            img_index += 1
        return res

    def register_prediction_metrics(self, inference_results):
        truth = {}
        predicts = {}
        row_uuids = []
        for head in self.metrics.heads:
            truth[head] = []
            predicts[head] = []
        for r in inference_results:
            image_id = json.loads(r["id"])
            row_uuids.append(image_id['id'])
            for head in self.metrics.heads:
                predicts[head].append(r[head]['index'])
                truth[head].append(image_id['label'][head])
        self.metrics.register_predictions_mutil_head(row_uuid=row_uuids, prediction=predicts, truth=truth)



    def evaluate(self)->Metrics:
        self.metrics.reset()
        batch_idx = 0
        batch_count = 0
        nbr_batches = len(self.test_data_set) // self.batch_size
        images = []
        targets = []
        pg = tqdm(total=nbr_batches, position=0, colour="blue",
                  desc='ONNX Evaluation')
        for image, target in self.test_data_set:
            images.append(image)
            targets.append(target)
            batch_idx += 1
            if batch_idx == self.batch_size:
                batch_count += 1
                pg.update()
                inference_batch = self.build_inference_batch(batch_count, images, targets)
                inference_result = self.inference.infer(inference_batch)
                self.register_prediction_metrics( inference_result)
                batch_idx = 0
                images = []
                targets = []

        self.metrics.register_performance_metrics(self.inference.get_performance_metrics())
        return self.metrics

