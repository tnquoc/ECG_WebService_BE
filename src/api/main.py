import os
import time

from fastapi import FastAPI

from src.api.schemas import ECGBeatInput
from src.models.beat_classification.inference import predict_ecg_beat_signals


app = FastAPI(
    title="ECG APP",
    summary="API for ECG APP",
    version="v0.1",
    docs_url="/docs",
    contact={
        "name": "Ngoc Quoc",
        "email": "tnquoc1998@gmail.com",
    },
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/ecg_beat_classification")
async def handle_ecg_beat_classification(item: ECGBeatInput):
    try:
        predicted_labels, predicted_symbols, scores = predict_ecg_beat_signals(item.ecg_data)
        return dict(
            error=0, 
            data=dict(
                predicted_beat_label=", ".join(predicted_labels), 
                predicted_beat_symbol=", ".join(predicted_symbols),
                score=", ".join(scores),
            )
        )
    except Exception as e:
        error_message = str(e)
        return dict(error=1, data={}, message=error_message)
