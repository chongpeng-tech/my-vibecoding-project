from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import warp_plate


def _augment_plate(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    if random.random() < 0.5:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-22, 22)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if random.random() < 0.4:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0)
    if random.random() < 0.3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.3), 0, 255).astype(np.uint8)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.8, 1.3), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


class CCPDOCRDataset(Dataset):
    def __init__(
        self,
        index_file: str | Path,
        image_width: int = 256,
        image_height: int = 64,
        augment: bool = False,
    ) -> None:
        self.index_file = Path(index_file)
        self.image_width = image_width
        self.image_height = image_height
        self.augment = augment
        self.samples = self._load_index(self.index_file)

    @staticmethod
    def _load_index(index_file: Path) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with index_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        corners = np.asarray(sample["corners"], dtype=np.float32)
        text = sample["plate_text"]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        plate = warp_plate(
            image_bgr=image,
            corners_rd_ld_lu_ru=corners,
            width=self.image_width,
            height=self.image_height,
        )
        if self.augment:
            plate = _augment_plate(plate)

        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(plate).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor, text


def ctc_collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    texts = [item[1] for item in batch]
    return images, texts

