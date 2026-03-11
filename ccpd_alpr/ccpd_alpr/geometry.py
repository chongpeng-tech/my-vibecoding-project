from __future__ import annotations

import cv2
import numpy as np


def rd_ld_lu_ru_to_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    if points.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {points.shape}")
    # CCPD order: right-down, left-down, left-up, right-up.
    return np.asarray([points[2], points[3], points[0], points[1]], dtype=np.float32)


def warp_plate(
    image_bgr: np.ndarray,
    corners_rd_ld_lu_ru: np.ndarray,
    width: int = 256,
    height: int = 64,
) -> np.ndarray:
    src = rd_ld_lu_ru_to_tl_tr_br_bl(corners_rd_ld_lu_ru.astype(np.float32))
    dst = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def bbox_xyxy_to_yolo(
    bbox_xyxy: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) * 0.5 / image_width
    cy = (y1 + y2) * 0.5 / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return cx, cy, w, h


def iou_xyxy(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

