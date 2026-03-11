from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import decode_plate_indices


@dataclass(slots=True)
class CCPDRecord:
    image_path: Path
    bbox_xyxy: tuple[int, int, int, int]
    corners_rd_ld_lu_ru: np.ndarray
    plate_indices: list[int]
    plate_text: str
    brightness: int
    blurriness: int


def _parse_xyxy(raw: str) -> tuple[int, int, int, int]:
    lu, rd = raw.split("_")
    x1, y1 = (int(v) for v in lu.split("&"))
    x2, y2 = (int(v) for v in rd.split("&"))
    return x1, y1, x2, y2


def _parse_corners(raw: str) -> np.ndarray:
    pts = []
    for point in raw.split("_"):
        x, y = (int(v) for v in point.split("&"))
        pts.append((x, y))
    if len(pts) != 4:
        raise ValueError(f"Expected 4 corners, got {len(pts)}")
    return np.asarray(pts, dtype=np.float32)


def _safe_int(raw: str, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def parse_ccpd_filename(image_path: str | Path) -> CCPDRecord:
    path = Path(image_path)
    parts = path.stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Invalid CCPD filename: {path.name}")

    bbox = _parse_xyxy(parts[2])
    corners = _parse_corners(parts[3])
    plate_indices = [_safe_int(v) for v in parts[4].split("_") if v != ""]
    plate_text = decode_plate_indices(plate_indices)
    brightness = _safe_int(parts[5], default=-1)
    blurriness = _safe_int(parts[6], default=-1)

    return CCPDRecord(
        image_path=path,
        bbox_xyxy=bbox,
        corners_rd_ld_lu_ru=corners,
        plate_indices=plate_indices,
        plate_text=plate_text,
        brightness=brightness,
        blurriness=blurriness,
    )


def parse_many(paths: Iterable[str | Path]) -> list[CCPDRecord]:
    records: list[CCPDRecord] = []
    for p in paths:
        try:
            records.append(parse_ccpd_filename(p))
        except ValueError:
            continue
    return records

