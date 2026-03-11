from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import province_charset


def _font_candidates() -> list[Path]:
    base = Path("C:/Windows/Fonts")
    names = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "simkai.ttf",
        "simfang.ttf",
        "STXihei.TTF",
    ]
    paths = [base / n for n in names if (base / n).exists()]
    return paths


def _render_char(char: str, font_path: Path, size: int, glyph_scale: float = 0.78) -> np.ndarray:
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    font_size = max(18, int(size * glyph_scale))
    font = ImageFont.truetype(str(font_path), font_size)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    img = np.array(canvas, dtype=np.uint8)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def _crop_candidates(plate_bgr: np.ndarray, size: int = 64) -> list[np.ndarray]:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x_ranges = [(0.01, 0.20), (0.03, 0.22), (0.00, 0.18)]
    y_ranges = [(0.05, 0.95), (0.10, 0.90)]
    crops: list[np.ndarray] = []
    for xr in x_ranges:
        for yr in y_ranges:
            x1 = int(w * xr[0])
            x2 = int(w * xr[1])
            y1 = int(h * yr[0])
            y2 = int(h * yr[1])
            roi = gray[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_CUBIC)
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            _, bin1 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bin2 = 255 - bin1
            crops.extend([bin1, bin2])
    return crops


def _score(template: np.ndarray, target: np.ndarray) -> float:
    # Both are size x size uint8, 0/255 binary.
    a = template.astype(np.float32) / 255.0
    b = target.astype(np.float32) / 255.0
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-6
    return float(np.sum(a * b) / denom)


@dataclass(slots=True)
class ProvinceRefineResult:
    best_char: str
    confidence: float
    top3: list[tuple[str, float]]
    replaced: bool
    original_char: str


class ProvinceRefiner:
    def __init__(self, size: int = 64) -> None:
        self.size = size
        self.provinces = province_charset()
        font_paths = _font_candidates()
        if not font_paths:
            raise RuntimeError("No Chinese fonts found under C:/Windows/Fonts")

        self.templates: dict[str, list[np.ndarray]] = {p: [] for p in self.provinces}
        scales = [0.72, 0.78, 0.84]
        for province in self.provinces:
            for fp in font_paths:
                for sc in scales:
                    try:
                        self.templates[province].append(_render_char(province, fp, size=size, glyph_scale=sc))
                    except Exception:
                        continue
            if not self.templates[province]:
                raise RuntimeError(f"Failed to render templates for province char: {province}")

    def predict(self, plate_bgr: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        crops = _crop_candidates(plate_bgr, size=self.size)
        if not crops:
            return "", 0.0, []

        scores: dict[str, float] = {}
        for province, templates in self.templates.items():
            best = -1.0
            for tmpl in templates:
                for crop in crops:
                    s = _score(tmpl, crop)
                    if s > best:
                        best = s
            scores[province] = best

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_char, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        confidence = max(0.0, best_score - second_score)
        return best_char, confidence, ranked[:3]

    def refine(
        self,
        plate_bgr: np.ndarray,
        plate_text: str,
        threshold: float = 0.08,
        force_when_invalid: bool = True,
    ) -> tuple[str, ProvinceRefineResult]:
        if not plate_text:
            empty = ProvinceRefineResult("", 0.0, [], False, "")
            return plate_text, empty

        original = plate_text[0]
        best_char, conf, top3 = self.predict(plate_bgr)
        replaced = False

        if not best_char:
            result = ProvinceRefineResult(best_char, conf, top3, False, original)
            return plate_text, result

        if force_when_invalid and original not in self.provinces and conf >= threshold:
            plate_text = best_char + plate_text[1:]
            replaced = True
        elif best_char != original and conf >= threshold:
            plate_text = best_char + plate_text[1:]
            replaced = True

        result = ProvinceRefineResult(best_char, conf, top3, replaced, original)
        return plate_text, result

