import os
import sys

import numpy as np
import torch

from src.models.ecg_interpretation.ecg_caption import TopicTransformer, Vocabulary

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


def predict_ecg_caption(input_signal):
    array = torch.from_numpy(np.array(input_signal)).type(torch.FloatTensor)
    array = array.unsqueeze(0).unsqueeze(0)
    words = caption_model.sample(array, max_length)
    generated = caption_model.vocab.decode(words, skip_first=False)
    caption = generated[0]

    return caption
