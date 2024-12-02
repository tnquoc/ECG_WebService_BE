import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import models

from src.utils.config import *

# load checkpoint
checkpoint_path = "src/models/beat_classification/checkpoint.h5"
model = models.load_model(checkpoint_path)


def preprocess(input_signals):
    processed_signals = []
    for signal in input_signals:
        raw_signal = [float(x) if x != 0.0 else 0.0 for x in signal]
        raw_signal = np.array(raw_signal)
        if len(raw_signal) < ECG_BEAT_LENGTH:
            raw_signal = np.pad(raw_signal, (0, ECG_BEAT_LENGTH - len(raw_signal)), "constant")
        else:
            raw_signal = raw_signal[:ECG_BEAT_LENGTH]
        processed_signals.append(raw_signal)
    return np.array(processed_signals).reshape(len(input_signals), ECG_BEAT_LENGTH, 1)


def predict_ecg_beat_signals(input_signals):
    try:
        ecg_data = preprocess(input_signals)

        # model prediction
        predictions = model(ecg_data)
        predictions = tf.nn.softmax(predictions, axis=1).numpy()

        # collect prediction
        predicted_labels = []
        predicted_symbols = []
        scores = []

        for prediction in predictions:
            predicted_class = np.argmax(prediction, axis=0)
            predicted_labels.append(ECG_BEAT_LABELS[predicted_class])
            predicted_symbols.append(ECG_BEAT_SYMBOLS[predicted_class])
            scores.append("{:.4f}".format(prediction[predicted_class]))

        return predicted_labels, predicted_symbols, scores

    except Exception as e:
        raise Exception(e)
