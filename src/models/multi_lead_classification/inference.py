import os

import numpy as np
import torch
from scipy.signal import resample

from src.models.multi_lead_classification.models import InceptionTime

from src.utils.config import (
    DATA_TYPE,
    ML_ECG_SIGNAL_LENGTH,
    ML_LABELS_LIST,
)
from src.utils.preprocess import (
    butter_bandpass_filter,
    baseline_wander_remove,
    normalize,
)

# load models
device = "cpu"
ckpt_path = (
    "src/models/multi_lead_classification/checkpoints/best_metric_checkpoint.pth"
)
checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
model = InceptionTime(in_channel=12, num_classes=5).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def preprocess_signal(signal):
    signal = np.array(signal)
    signal.astype(DATA_TYPE)
    signal = butter_bandpass_filter(signal, 1, 30, 250, order=2)
    signal = baseline_wander_remove(signal, 250, 0.2, 0.6)
    signal = normalize(signal, int(0.5 * 250))
    return signal


def predict_ml_ecg_signals(input_signals):
    try:
        # preprocessed_signals = [preprocess_signal(signal) for signal in input_signals]

        # Resample each segment from 250 Hz to 100 Hz
        resampled_signals = [
            # resample(segment, ML_ECG_SIGNAL_LENGTH) for segment in preprocessed_signals
            resample(segment, ML_ECG_SIGNAL_LENGTH)
            for segment in input_signals
        ]

        ecg_data = np.array(resampled_signals)
        ecg_data = torch.from_numpy(ecg_data).type(torch.FloatTensor).unsqueeze(0)

        # Model prediction
        outputs = model(ecg_data.to(device))
        predictions = torch.sigmoid(outputs).cpu().detach().numpy()[0]

        # Collect prediction results
        predicted_labels = []

        for i in range(len(predictions)):
            if predictions[i] > 0.5:
                predicted_labels.append(ML_LABELS_LIST[i])

        return str(predicted_labels)

    except Exception as e:
        raise Exception(e)
