import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import ECGBeatInput, ECGSignalInput
from src.models.beat_classification.inference import (
    preprocess_signal,
    predict_ecg_beat_signals,
    qrs_detection,
    extract_qrs_segments,
)


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


# Add CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/ecg_beat_classification")
async def handle_ecg_beat_classification(item: ECGBeatInput):
    try:
        predicted_labels, predicted_symbols, scores = predict_ecg_beat_signals(
            item.ecg_data
        )
        return dict(
            error=0,
            data=dict(
                predicted_beat_label=", ".join(predicted_labels),
                predicted_beat_symbol=", ".join(predicted_symbols),
                score=", ".join(scores),
            ),
        )
    except Exception as e:
        error_message = str(e)
        return dict(error=1, data={}, message=error_message)


@app.post("/ecg_beat_analysis")
async def handle_ecg_beat_analysis(item: ECGSignalInput):
    try:
        # Process ECG signal
        signal = item.ecg
        processed_signal = preprocess_signal(signal)
        # processed_signal = signal

        # detect qrs beat
        qrs_beat_positions, qrs_ind_beat = qrs_detection(processed_signal)

        # extract list ecg signal
        qrs_segments = extract_qrs_segments(processed_signal, qrs_beat_positions)

        # detect beats labels
        predicted_labels, predicted_symbols, scores = predict_ecg_beat_signals(
            qrs_segments
        )

        return dict(
            error=0,
            data=dict(
                predicted_beat_label=", ".join(predicted_labels),
                predicted_beat_symbol=predicted_symbols,
                qrs_beat_positions=qrs_beat_positions.tolist(),
                qrs_ind_beat=qrs_ind_beat.tolist(),
                # qrs_ind_beat=qrs_segments.tolist(),
            ),
        )
    except Exception as e:
        error_message = str(e)
        return dict(error=1, data={}, message=error_message)
