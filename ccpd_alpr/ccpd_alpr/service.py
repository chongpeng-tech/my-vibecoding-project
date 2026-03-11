from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from .constants import province_charset
from .geometry import warp_plate
from .pipeline import load_recognizer, recognize_plate
from .province_classifier import ProvinceClassifier
from .province_refiner import ProvinceRefiner
from .utils import ensure_dir


@dataclass(slots=True)
class ALPRPrediction:
    detected: bool
    det_conf: float
    bbox_xyxy: tuple[float, float, float, float] | None
    corners_rd_ld_lu_ru: np.ndarray | None
    plate_text: str
    ocr_conf: float
    plate_crop_bgr: np.ndarray | None
    annotated_bgr: np.ndarray
    province_refined: bool = False
    province_candidate: str = ""
    province_conf: float = 0.0
    error: str = ""

    def to_record(self, image_path: str | None = None, frame_idx: int | None = None) -> dict[str, Any]:
        return {
            "image_path": image_path or "",
            "frame_idx": frame_idx if frame_idx is not None else -1,
            "detected": self.detected,
            "det_conf": float(self.det_conf),
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "corners_rd_ld_lu_ru": self.corners_rd_ld_lu_ru.tolist() if self.corners_rd_ld_lu_ru is not None else None,
            "plate_text": self.plate_text,
            "ocr_conf": float(self.ocr_conf),
            "province_refined": bool(self.province_refined),
            "province_candidate": self.province_candidate,
            "province_conf": float(self.province_conf),
            "error": self.error,
        }


def _collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    images = [p for p in source.rglob("*") if p.is_file() and p.suffix in exts]
    images.sort()
    return images


def _draw_prediction(image_bgr: np.ndarray, pred: ALPRPrediction) -> np.ndarray:
    vis = image_bgr.copy()
    if pred.bbox_xyxy is not None:
        x1, y1, x2, y2 = (int(v) for v in pred.bbox_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 125, 0), 2)
    if pred.corners_rd_ld_lu_ru is not None:
        pts = pred.corners_rd_ld_lu_ru.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
        for p in pred.corners_rd_ld_lu_ru.astype(np.int32):
            cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 165, 255), -1, lineType=cv2.LINE_AA)

    if pred.detected:
        refine_tag = " *P" if pred.province_refined else ""
        caption = f"{pred.plate_text}{refine_tag}  det:{pred.det_conf:.2f}  ocr:{pred.ocr_conf:.2f}"
        y = 28
        if pred.bbox_xyxy is not None:
            y = max(24, int(pred.bbox_xyxy[1]) - 8)
        cv2.putText(vis, caption, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 240, 250), 2, cv2.LINE_AA)
    elif pred.error:
        cv2.putText(vis, pred.error[:80], (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 255), 2, cv2.LINE_AA)

    return vis


def _sanitize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.nan_to_num(corners.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out[:, 0] = np.clip(out[:, 0], 0, max(width - 1, 1))
    out[:, 1] = np.clip(out[:, 1], 0, max(height - 1, 1))
    return out


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class ALPRService:
    def __init__(
        self,
        detector_weights: str | Path,
        recognizer_weights: str | Path,
        device: str = "cuda:0",
        province_refine: bool = True,
        province_refine_threshold: float = 0.26,
        province_classifier_weights: str | Path = "runs/province_classifier/best.pt",
    ) -> None:
        self.detector_weights = Path(detector_weights)
        self.recognizer_weights = Path(recognizer_weights)
        self.device = device

        if not self.detector_weights.exists():
            raise FileNotFoundError(f"Detector weights not found: {self.detector_weights}")
        if not self.recognizer_weights.exists():
            raise FileNotFoundError(f"Recognizer weights not found: {self.recognizer_weights}")

        self.detector = YOLO(self.detector_weights.as_posix())
        self.recog_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.recognizer, self.converter = load_recognizer(self.recognizer_weights, device=self.recog_device)
        self.province_refine_threshold = float(province_refine_threshold)
        self.province_chars = set(province_charset())
        self.province_classifier: ProvinceClassifier | None = None
        self.province_refiner: ProvinceRefiner | None = None
        if province_refine:
            clf_path = Path(province_classifier_weights)
            if clf_path.exists():
                try:
                    self.province_classifier = ProvinceClassifier(clf_path, device=self.recog_device)
                except Exception:
                    self.province_classifier = None
            if self.province_classifier is None:
                # Fallback only when classifier is absent.
                self.province_refiner = ProvinceRefiner()

    def predict_image(
        self,
        image_bgr: np.ndarray,
        imgsz: int = 960,
        conf: float = 0.25,
        iou: float = 0.6,
        max_det: int = 1,
    ) -> ALPRPrediction:
        if image_bgr is None:
            raise ValueError("Input image is None")
        height, width = image_bgr.shape[:2]

        result = self.detector.predict(
            source=image_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device,
            verbose=False,
        )[0]

        pred = ALPRPrediction(
            detected=False,
            det_conf=0.0,
            bbox_xyxy=None,
            corners_rd_ld_lu_ru=None,
            plate_text="",
            ocr_conf=0.0,
            plate_crop_bgr=None,
            annotated_bgr=image_bgr.copy(),
            province_refined=False,
            province_candidate="",
            province_conf=0.0,
            error="未检测到车牌",
        )

        if result.boxes is None or len(result.boxes) == 0:
            pred.annotated_bgr = _draw_prediction(image_bgr, pred)
            return pred
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            pred.error = "检测到框，但未检测到角点"
            pred.annotated_bgr = _draw_prediction(image_bgr, pred)
            return pred

        box = result.boxes.xyxy[0].detach().cpu().numpy().astype(float).tolist()
        det_conf = float(result.boxes.conf[0].detach().cpu().item()) if result.boxes.conf is not None else 0.0
        corners = result.keypoints.xy[0].detach().cpu().numpy().astype(np.float32)
        corners = _sanitize_corners(corners, width=width, height=height)

        pred.bbox_xyxy = tuple(float(v) for v in box)
        pred.det_conf = det_conf
        pred.corners_rd_ld_lu_ru = corners

        if _polygon_area(corners) < 12.0:
            pred.error = "角点异常，跳过识别"
            pred.annotated_bgr = _draw_prediction(image_bgr, pred)
            return pred

        plate_crop = warp_plate(image_bgr, corners, width=256, height=64)
        plate_text, ocr_conf = recognize_plate(self.recognizer, self.converter, plate_crop, self.recog_device)
        province_refined = False
        province_candidate = ""
        province_conf = 0.0
        if self.province_classifier is not None and plate_text:
            province_pred = self.province_classifier.predict(plate_crop)
            province_candidate = province_pred.char
            province_conf = float(province_pred.conf)
            province_margin = (
                float(province_pred.top3[0][1] - province_pred.top3[1][1])
                if len(province_pred.top3) > 1
                else province_conf
            )
            first = plate_text[0]
            replace = False
            if first not in self.province_chars and province_conf >= self.province_refine_threshold:
                replace = True
            elif (
                province_candidate != first
                and province_conf >= (self.province_refine_threshold + 0.08)
                and province_margin >= 0.04
            ):
                replace = True
            if replace:
                plate_text = province_candidate + plate_text[1:]
                province_refined = True
        elif self.province_refiner is not None and plate_text:
            refined_text, refine_info = self.province_refiner.refine(
                plate_bgr=plate_crop,
                plate_text=plate_text,
                threshold=self.province_refine_threshold,
                force_when_invalid=True,
            )
            plate_text = refined_text
            province_refined = refine_info.replaced
            province_candidate = refine_info.best_char
            province_conf = float(refine_info.confidence)

        pred.detected = True
        pred.error = ""
        pred.plate_crop_bgr = plate_crop
        pred.plate_text = plate_text
        pred.ocr_conf = float(ocr_conf)
        pred.province_refined = province_refined
        pred.province_candidate = province_candidate
        pred.province_conf = province_conf
        pred.annotated_bgr = _draw_prediction(image_bgr, pred)
        return pred

    def infer_directory(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        imgsz: int = 960,
        conf: float = 0.25,
        iou: float = 0.6,
        max_det: int = 1,
    ) -> dict[str, Any]:
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source}")
        images = _collect_images(source)
        if not images:
            raise RuntimeError(f"No images found in {source}")

        out_root = ensure_dir(output_dir)
        vis_dir = ensure_dir(out_root / "vis")
        records: list[dict[str, Any]] = []

        for image_path in images:
            image = cv2.imread(image_path.as_posix())
            if image is None:
                continue
            pred = self.predict_image(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det)
            cv2.imwrite((vis_dir / image_path.name).as_posix(), pred.annotated_bgr)
            records.append(pred.to_record(image_path=image_path.resolve().as_posix()))

        jsonl_path = out_root / "predictions.jsonl"
        csv_path = out_root / "predictions.csv"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pd.DataFrame.from_records(records).to_csv(csv_path, index=False, encoding="utf-8-sig")

        detected = sum(1 for r in records if r["detected"])
        return {
            "num_images": len(records),
            "num_detected": detected,
            "detect_rate": detected / max(len(records), 1),
            "vis_dir": vis_dir.resolve().as_posix(),
            "jsonl_path": jsonl_path.resolve().as_posix(),
            "csv_path": csv_path.resolve().as_posix(),
            "records": records,
        }

    def infer_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        imgsz: int = 960,
        conf: float = 0.25,
        iou: float = 0.6,
        max_det: int = 1,
        frame_step: int = 1,
    ) -> dict[str, Any]:
        source = Path(video_path)
        if not source.exists():
            raise FileNotFoundError(f"Video path not found: {source}")

        out_root = ensure_dir(output_dir)
        video_out = out_root / f"{source.stem}_annotated.mp4"
        jsonl_path = out_root / f"{source.stem}_predictions.jsonl"
        csv_path = out_root / f"{source.stem}_predictions.csv"

        cap = cv2.VideoCapture(source.as_posix())
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            video_out.as_posix(),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video writer: {video_out}")

        rows: list[dict[str, Any]] = []
        last_pred: ALPRPrediction | None = None
        idx = 0
        step = max(int(frame_step), 1)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0 or last_pred is None:
                pred = self.predict_image(frame, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det)
                last_pred = pred
            else:
                pred = last_pred
                pred = ALPRPrediction(
                    detected=pred.detected,
                    det_conf=pred.det_conf,
                    bbox_xyxy=pred.bbox_xyxy,
                    corners_rd_ld_lu_ru=None if pred.corners_rd_ld_lu_ru is None else pred.corners_rd_ld_lu_ru.copy(),
                    plate_text=pred.plate_text,
                    ocr_conf=pred.ocr_conf,
                    plate_crop_bgr=None if pred.plate_crop_bgr is None else pred.plate_crop_bgr.copy(),
                    annotated_bgr=_draw_prediction(frame, pred),
                    error=pred.error,
                )
            writer.write(pred.annotated_bgr)
            rows.append(pred.to_record(image_path=source.resolve().as_posix(), frame_idx=idx))
            idx += 1

        cap.release()
        writer.release()

        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pd.DataFrame.from_records(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

        detected = sum(1 for r in rows if r["detected"])
        return {
            "num_frames": len(rows),
            "fps": fps,
            "frame_count_meta": frame_count,
            "num_detected": detected,
            "detect_rate": detected / max(len(rows), 1),
            "video_out": video_out.resolve().as_posix(),
            "jsonl_path": jsonl_path.resolve().as_posix(),
            "csv_path": csv_path.resolve().as_posix(),
            "records": rows,
        }
