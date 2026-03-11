from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .recognizer_model import CRNNRecognizer
from .tokenizer import CTCLabelConverter


def preprocess_plate_for_ocr(plate_bgr: np.ndarray) -> torch.Tensor:
    plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(plate_rgb).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


def load_recognizer(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CRNNRecognizer, CTCLabelConverter]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    charset = ckpt["charset"]
    converter = CTCLabelConverter(charset=charset)
    model = CRNNRecognizer(num_classes=converter.num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, converter


@torch.inference_mode()
def recognize_plate(
    model: CRNNRecognizer,
    converter: CTCLabelConverter,
    plate_bgr: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[str, float]:
    x = preprocess_plate_for_ocr(plate_bgr).to(device)
    logits = model(x)
    log_probs = logits.log_softmax(dim=2)
    text = converter.decode_batch(log_probs)[0]
    probs = log_probs.exp().max(dim=2).values.mean().item()
    return text, float(probs)
