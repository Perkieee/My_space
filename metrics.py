from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report

from lib.trainer.constant import MetricsType
from lib.trainer.constant import PredictionFields
from lib.trainer.constant import TrainingStep
import json

METRIC_TRAIN_ACCURACY = MetricsType.TRAIN_ACC
METRIC_TRAIN_LOSS = MetricsType.TRAIN_LOSS
METRIC_VAL_ACCURACY = MetricsType.VAL_ACC
METRIC_VAL_LOSS = MetricsType.VAL_LOSS

PRED_UUID = PredictionFields.UUID
PRED_PREDICTIONS = PredictionFields.PREDICTIONS
PRED_TRUTH = PredictionFields.TRUTH
PRED_CORRECT = PredictionFields.CORRECT


def new_history():
    return dict({METRIC_TRAIN_ACCURACY: [],
                 METRIC_TRAIN_LOSS: [],
                 METRIC_VAL_ACCURACY: [],
                 METRIC_VAL_LOSS: []})


def new_prediction():
    return dict({PRED_UUID: [],
                 PRED_PREDICTIONS: [],
                 PRED_TRUTH: [],
                 PRED_CORRECT: []})


def new_evaluation():
    return dict({METRIC_VAL_ACCURACY: [],
                 METRIC_VAL_LOSS: []})

def new_prediction_multi_head(head_names: list[str]):
    p={
        PRED_UUID: []
    }
    for head in head_names:
       p[head] = new_prediction()
    return p

class Metrics(object):

    def __init__(self):
        self.train_history = None
        self.predictions = None
        self.epoch_history = None
        self.step = None
        self.epoch = 0
        self.heads=None
        self.performance=None

    def reset(self):
        self.epoch_history = new_history()
        self.train_history = new_history()
        if self.heads is not None:
            self.predictions = new_prediction_multi_head(self.heads)
        else:
           self.predictions = new_prediction()

    def end_train(self):
        self.__save_epoch_history()

    def __save_epoch_history(self):
        if len(self.epoch_history[METRIC_TRAIN_LOSS]) > 0:
            accuracy = self.__mean(self.epoch_history, METRIC_TRAIN_ACCURACY)
            self.train_history[METRIC_TRAIN_ACCURACY].append(accuracy)
            loss = self.__mean(self.epoch_history, METRIC_TRAIN_LOSS)
            self.train_history[METRIC_TRAIN_LOSS].append(loss)
        if len(self.epoch_history[METRIC_VAL_LOSS]) > 0:
            accuracy = self.__mean(self.epoch_history, METRIC_VAL_ACCURACY)
            self.train_history[METRIC_VAL_ACCURACY].append(accuracy)
            loss = self.__mean(self.epoch_history, METRIC_VAL_LOSS)
            self.train_history[METRIC_VAL_LOSS].append(loss)

    def save_history(self,file_path):
        with open(file_path, 'w') as outfile:
            json.dump(self.train_history, outfile)

    def load_history_training(self,file_path):
        with open(file_path, 'r') as infile:
            self.train_history = json.load(infile)

    def new_epoch(self):
        self.epoch = +1
        self.__save_epoch_history()
        self.epoch_history = new_history()

    def register_batch(self, acc, loss):
        if self.step == TrainingStep.TRAIN_EPOCH:
            self.epoch_history[METRIC_TRAIN_ACCURACY].append(acc)
            self.epoch_history[METRIC_TRAIN_LOSS].append(loss)
        if self.step == TrainingStep.VAL_EPOCH or self.step == TrainingStep.EVALUATION:
            self.epoch_history[METRIC_VAL_ACCURACY].append(acc)
            self.epoch_history[METRIC_VAL_LOSS].append(loss)

    @staticmethod
    def __mean(bag: dict, metrics_id: MetricsType):
        if bag is None or len(bag[metrics_id]) == 0:
            return 0.0
        return np.mean(bag[metrics_id]).mean()

    def accuracy(self):
        if self.step == TrainingStep.TRAIN_EPOCH:
            return self.__mean(self.epoch_history, METRIC_TRAIN_ACCURACY)
        if self.step == TrainingStep.VAL_EPOCH or self.step == TrainingStep.EVALUATION:
            return self.__mean(self.epoch_history, METRIC_VAL_ACCURACY)

    def loss(self):
        if self.step == TrainingStep.TRAIN_EPOCH:
            return self.__mean(self.epoch_history, METRIC_TRAIN_LOSS)
        if self.step == TrainingStep.VAL_EPOCH or self.step == TrainingStep.EVALUATION:
            return self.__mean(self.epoch_history, METRIC_VAL_LOSS)
        return None

    @staticmethod
    def __correct(l1: list[int], l2: list[int]) -> list[bool]:
        return [i == j for i, j in zip(l1, l2)]

    def register_predictions(self, row_uuid: Optional[list], prediction: list[int], truth: Optional[list[int]] = None):
        if self.heads is not None:
            raise Exception("register_predictions_mutil_head should be used")
        if row_uuid is not None:
            self.predictions[PRED_UUID].extend(row_uuid)
        self.predictions[PRED_PREDICTIONS].extend(prediction)
        if truth is not None:
            self.predictions[PRED_TRUTH].extend(truth)
            self.predictions[PRED_CORRECT].extend(self.__correct(prediction, truth))


    def register_predictions_mutil_head(self,row_uuid: Optional[list], prediction: dict[str, list[int]], truth: Optional[dict[str, list[int]]] = None):
        if self.heads is None:
           raise Exception("register_predictions should be used")
        if row_uuid is not None:
            self.predictions[PRED_UUID].extend(row_uuid)
        for head in self.heads:
            self.predictions[head][PRED_PREDICTIONS].extend(prediction[head])
        if truth is not None:
            for head in prediction.keys():
                self.predictions[head][PRED_TRUTH].extend(truth[head])
                self.predictions[head][PRED_CORRECT].extend(self.__correct(prediction[head], truth[head]))


    def get_predictions(self):
        if self.heads is not None:
            raise ValueError("get_predictions_mutil_head should be used")
        if self.predictions is None:
            return new_prediction()
        if len(self.predictions[PRED_TRUTH]) == 0:
            del self.predictions[PRED_TRUTH]
            del self.predictions[PRED_CORRECT]
        if len(self.predictions[PRED_UUID]) == 0:
            del self.predictions[PRED_UUID]
        return self.predictions

    def get_predictions_mutil_head(self):
        if self.heads is None:
            raise ValueError("get_predictions should be used")
        if self.predictions is None:
            return new_prediction_multi_head(self.heads)
        for head in self.heads:
            pred_head = self.predictions[head]
            if len(pred_head[PRED_TRUTH]) == 0:
                del(pred_head[PRED_TRUTH])
                del(pred_head[PRED_CORRECT])
            if len(pred_head[PRED_UUID]) == 0:
                del(pred_head[PRED_UUID])
            self.predictions[head] = pred_head
        return self.predictions


    def get_evaluation_metric(self):
        return {METRIC_VAL_LOSS: self.loss(),
                METRIC_VAL_ACCURACY: self.accuracy()}

    def register_performance_metrics(self,performance:dict[str,float]):
        self.performance = performance

    def get_performance_metrics(self):
        return self.performance



def plot_model_history(name, history, nb_epochs):
    epochs = [i for i in range(nb_epochs)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history[METRIC_TRAIN_ACCURACY]
    train_loss = history[METRIC_TRAIN_LOSS]
    val_acc = history[METRIC_VAL_ACCURACY]
    val_loss = history[METRIC_VAL_LOSS]
    fig.set_size_inches(16, 9)

    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'ro-', label='Val Accuracy')
    ax[0].set_title('Training & Validation Accuracy for tasks ' + name)
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r-o', label='Val Loss')
    ax[1].set_title('Training & Validation Loss for the tasks ' + name)
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.show()


def plot_confusion_matrix(predictions):
    cm = confusion_matrix(predictions[PRED_PREDICTIONS], predictions[PRED_TRUTH])
    sn.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def get_classification_report(predictions):
    return classification_report(predictions[PRED_PREDICTIONS], predictions[PRED_TRUTH])
