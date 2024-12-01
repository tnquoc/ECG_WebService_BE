from fastapi import FastAPI
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
from tensorflow.keras import models

app = FastAPI()

# load checkpoint

checkpoint_path = "logs/best_loss_checkpoint.h5"
model = models.load_model(checkpoint_path)

signal_labels = ["Normal", "LBBB", "RBBB", "PVC", "APB"]
signal_symbol = ["N", "L", "R", "V", "A"]


class ECGInput(BaseModel):
    ecg_data: list[list[float]] = Field(
        ...,
        description="Danh sách các tín hiệu ECG, mỗi tín hiệu là một danh sách 320 giá trị.",
    )

    def preprocess(self):
        processed_signals = []
        for signal in self.ecg_data:
            raw_signal = [float(x) if x != 0.0 else 0.0 for x in signal]
            raw_signal = np.array(raw_signal)
            if len(raw_signal) < 320:
                raw_signal = np.pad(raw_signal, (0, 320 - len(raw_signal)), "constant")
            else:
                raw_signal = raw_signal[:320]
            processed_signals.append(raw_signal)
        return np.array(processed_signals).reshape(len(self.ecg_data), 320, 1)


@app.post("/ecg_beats_classification")
def predict_ecg_signals(input_signals: ECGInput):
    try:
        ecg_data = input_signals.preprocess()

        # predict signal
        predictions = model(ecg_data)
        predictions = tf.nn.softmax(predictions, axis=1).numpy()

        # Dự đoán cho từng tín hiệu
        predicted_labels = []
        predicted_symbols = []
        scores = []

        for prediction in predictions:
            predicted_class = np.argmax(prediction, axis=0)
            predicted_labels.append(signal_labels[predicted_class])
            predicted_symbols.append(signal_symbol[predicted_class])
            scores.append("{:.4f}".format(prediction[predicted_class]))

        # Trả về kết quả
        return {
            "status": "success",
            "predicted_beat_label": ", ".join(predicted_labels),
            "predicted_beat_symbol": ", ".join(predicted_symbols),
            "score": ", ".join(scores),
        }

    except Exception as e:
        # Trả thông báo lỗi nếu có
        return {"status": "error", "message": str(e)}
