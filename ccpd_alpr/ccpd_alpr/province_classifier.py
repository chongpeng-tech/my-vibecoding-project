from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .constants import province_charset


def crop_first_char_region(plate_bgr: np.ndarray, size: int = 64) -> np.ndarray:
    h, w = plate_bgr.shape[:2]
    x1 = int(w * 0.01)
    x2 = int(w * 0.22)
    y1 = int(h * 0.04)
    y2 = int(h * 0.96)
    roi = plate_bgr[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)]
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return gray


class ProvinceClassifierNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.classifier(feat)


@dataclass(slots=True)
class ProvincePrediction:
    char: str
    conf: float
    top3: list[tuple[str, float]]


class ProvinceClassifier:
    def __init__(self, weights: str | Path, device: str | torch.device = "cpu") -> None:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        self.charset: list[str] = ckpt["charset"]
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.model = ProvinceClassifierNet(num_classes=len(self.charset))
        self.model.load_state_dict(ckpt["model_state"])
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, plate_bgr: np.ndarray) -> ProvincePrediction:
        gray = crop_first_char_region(plate_bgr, size=64)
        x = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        x = x.to(self.device)
        logits = self.model(x)
        probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()
        order = np.argsort(-probs)
        best_idx = int(order[0])
        best_char = self.charset[best_idx]
        top3 = [(self.charset[int(i)], float(probs[int(i)])) for i in order[:3]]
        return ProvincePrediction(char=best_char, conf=float(probs[best_idx]), top3=top3)


def default_province_charset() -> list[str]:
    return province_charset()

