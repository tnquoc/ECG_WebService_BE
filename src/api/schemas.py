from pydantic import BaseModel, Field


class ECGBeatInput(BaseModel):
    ecg_data: list[list[float]] = Field(
        ...,
        description="List of ECG signals, each signal is a list of 320 values.",
    )