import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from scipy.signal import resample

from src.utils.config import (
    ECG_BEAT_SYMBOLS,
    ECG_BEAT_LENGTH,
    ECG_BEAT_LABELS,
    FREQUENCY_SAMPLING,
    DATA_TYPE,
)
from src.utils.preprocess import (
    butter_bandpass_filter,
    baseline_wander_remove,
    normalize,
)

# load models
beat_cls_ckpt_path = "src/models/beat_classification/checkpoints/beat_cls_ckpt.h5"
beat_cls_model = models.load_model(beat_cls_ckpt_path)

qrs_detection_ckpt_path = (
    "src/models/beat_classification/checkpoints/qrs_detection_ckpt.h5"
)
qrs_detection_model = models.load_model(qrs_detection_ckpt_path)


def preprocess_beat_signals(input_signals):
    processed_signals = []
    for signal in input_signals:
        raw_signal = [float(x) if x != 0.0 else 0.0 for x in signal]
        raw_signal = np.array(raw_signal)
        if len(raw_signal) < ECG_BEAT_LENGTH:
            raw_signal = np.pad(
                raw_signal, (0, ECG_BEAT_LENGTH - len(raw_signal)), "constant"
            )
        else:
            raw_signal = raw_signal[:ECG_BEAT_LENGTH]
        processed_signals.append(raw_signal)
    return np.array(processed_signals).reshape(len(input_signals), ECG_BEAT_LENGTH, 1)


def predict_ecg_beat_signals(input_signals):
    try:
        # Resample each segment from 250 Hz to 360 Hz
        resampled_signals = [
            resample(segment, ECG_BEAT_LENGTH) for segment in input_signals
        ]
        ecg_data = np.array(resampled_signals).reshape(
            len(input_signals), ECG_BEAT_LENGTH, 1
        )

        # Optional: Plot resampled signals for debugging
        # for i, signal in enumerate(ecg_data):
        #     plt.figure()
        #     plt.plot(signal)
        #     plt.savefig(f"src/images/ecg_beat_classification_{i}.png")

        # Model prediction
        predictions = beat_cls_model(ecg_data, training=False)
        predictions = tf.nn.softmax(predictions, axis=1).numpy()

        # Collect prediction results
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


def preprocess_signal(signal):
    signal = np.array(signal)
    signal.astype(DATA_TYPE)
    signal = butter_bandpass_filter(signal, 1, 30, 250, order=2)
    signal = baseline_wander_remove(signal, 250, 0.2, 0.6)
    signal = normalize(signal, int(0.5 * 250))
    return signal


def clustering(data):
    positive_point = np.where(data == 1)[0]
    beat = []
    if len(positive_point) > 5:
        cluster = np.array([positive_point[0]])
        for i in range(1, len(positive_point)):
            if (
                positive_point[i] - cluster[-1] > 0.08 * FREQUENCY_SAMPLING
                or i == len(positive_point) - 1
            ):
                if i == len(positive_point) - 1:
                    cluster = np.append(cluster, positive_point[i])
                if cluster.shape[0] > 5:
                    beat.append(int(np.mean(cluster)))
                cluster = np.array([positive_point[i]])
            else:
                cluster = np.append(cluster, positive_point[i])

    return np.asarray(beat)


def qrs_detection(signal):
    ind = np.arange(101)[None, :] + np.arange(len(signal) - 101)[:, None]
    input = signal[ind][:, :, np.newaxis]
    output = qrs_detection_model.predict(input)
    result = np.argmax(output, axis=1)
    beat_final = clustering(result) + 25
    ind_beats = np.flatnonzero(result) + 25

    return beat_final, ind_beats


def extract_qrs_segments(signal, beat_final, segment_length=320):
    aligned_segments = []
    half_length = segment_length // 2
    signal_len = len(signal)

    for beat in beat_final:
        start = beat - half_length
        end = beat + half_length

        # Initialize segment with signal slice within bounds
        segment = signal[max(start, 0) : min(end, signal_len)]

        # Pad before if start < 0
        if start < 0:
            pad_start = -start
            pre_pad = signal[0:pad_start][::-1]  # reflect from start
            segment = np.concatenate((pre_pad, segment))

        # Pad after if end > signal_len
        if end > signal_len:
            pad_end = end - signal_len
            post_pad = signal[-pad_end:][::-1]  # reflect from end
            segment = np.concatenate((segment, post_pad))

        # If still short due to signal being very short
        if len(segment) < segment_length:
            # Final sanity pad with signal[0] or signal[-1] if needed
            segment = np.pad(segment, (0, segment_length - len(segment)), mode="edge")

        aligned_segments.append(segment[:segment_length])  # ensure exact length

    return np.array(aligned_segments)
