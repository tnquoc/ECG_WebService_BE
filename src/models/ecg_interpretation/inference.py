import os
import sys

import numpy as np
import torch
import tensorflow as tf

from src.models.ecg_interpretation.ecg_caption import TopicTransformer, Vocabulary
from src.models.ecg_interpretation.hr_model import load_hr_model

# sys.modules["__main__"] = sys.modules[__name__]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# load caption model
sys.modules["__main__"].Vocabulary = Vocabulary
caption_model = TopicTransformer.load_from_checkpoint(
    checkpoint_path="src/models/ecg_interpretation/checkpoints/ecg_caption.ckpt"
)
caption_model.eval()
max_length = 50


# load HR model
hr_model = load_hr_model(
    checkpoint_dir="src/models/ecg_interpretation/checkpoints/HR_model_best_MAE"
)


def predict_ecg_caption(input_signal):
    array = torch.from_numpy(np.array(input_signal)).type(torch.FloatTensor)
    array = array.unsqueeze(0).unsqueeze(0)
    words = caption_model.sample(array, max_length)
    generated = caption_model.vocab.decode(words, skip_first=False)
    caption = generated[0]

    return caption


def enhance_ecg_caption(input_signal, caption):
    data_for_hr_model = input_signal.T
    data_for_hr_model = np.expand_dims(data_for_hr_model, 0)
    with tf.device("/cpu:0"):
        predict_hr_1 = hr_model.predict(data_for_hr_model[:, :2500])[0, 0]
        predict_hr_2 = hr_model.predict(data_for_hr_model[:, 2500:])[0, 0]
    final_predict_hr = int((predict_hr_1 + predict_hr_2) // 2)
    print("HR", final_predict_hr, predict_hr_1, predict_hr_2)
    if final_predict_hr > 10:
        caption = caption.replace("<unk> bpm", f"{final_predict_hr} bpm")

    return caption


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
