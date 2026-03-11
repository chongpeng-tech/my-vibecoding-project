from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ccpd_alpr.geometry import iou_xyxy, warp_plate
from ccpd_alpr.pipeline import load_recognizer, recognize_plate
from ccpd_alpr.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end CCPD ALPR pipeline.")
    parser.add_argument(
        "--test-index",
        type=Path,
        default=Path("data/index/test.jsonl"),
        help="Path to test split jsonl.",
    )
    parser.add_argument("--detector-weights", type=Path, required=True, help="YOLO pose checkpoint.")
    parser.add_argument("--recognizer-weights", type=Path, required=True, help="CRNN checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("runs/eval/e2e_metrics.json"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    return parser.parse_args()


def load_samples(index_file: Path) -> list[dict]:
    samples: list[dict] = []
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output.parent)

    samples = load_samples(args.test_index)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No test samples were loaded.")

    detector = YOLO(args.detector_weights.as_posix())
    reco_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    recognizer, converter = load_recognizer(args.recognizer_weights, device=reco_device)

    total = len(samples)
    det_found = 0
    det_iou_ok = 0
    iou_sum = 0.0
    kpt_nrmse_sum = 0.0
    e2e_exact_all = 0
    e2e_exact_detected = 0
    e2e_exact_loc_ok = 0

    for sample in tqdm(samples, desc="e2e-eval"):
        image = cv2.imread(sample["image_path"])
        if image is None:
            continue

        result = detector.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=1,
            device=args.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            continue

        det_found += 1
        pred_box = tuple(float(v) for v in result.boxes.xyxy[0].cpu().numpy().tolist())
        gt_box = tuple(float(v) for v in sample["bbox"])
        current_iou = iou_xyxy(pred_box, gt_box)
        iou_sum += current_iou

        pred_kpts = result.keypoints.xy[0].cpu().numpy().astype(np.float32)
        gt_kpts = np.asarray(sample["corners"], dtype=np.float32)
        norm = np.linalg.norm(
            [
                max(image.shape[1], 1),
                max(image.shape[0], 1),
            ]
        )
        kpt_nrmse_sum += float(np.sqrt(np.mean((pred_kpts - gt_kpts) ** 2)) / (norm + 1e-6))

        if current_iou >= args.iou_threshold:
            det_iou_ok += 1

        plate_patch = warp_plate(image, pred_kpts, width=256, height=64)
        pred_text, _ = recognize_plate(recognizer, converter, plate_patch, reco_device)
        gt_text = sample["plate_text"]

        if pred_text == gt_text:
            e2e_exact_all += 1
            e2e_exact_detected += 1
            if current_iou >= args.iou_threshold:
                e2e_exact_loc_ok += 1

    metrics = {
        "num_samples": total,
        "detected": det_found,
        "det_recall": det_found / max(total, 1),
        f"det_recall_iou{args.iou_threshold:.2f}": det_iou_ok / max(total, 1),
        "mean_iou_on_detected": iou_sum / max(det_found, 1),
        "mean_corner_nrmse_on_detected": kpt_nrmse_sum / max(det_found, 1),
        "e2e_exact_all": e2e_exact_all / max(total, 1),
        "e2e_exact_on_detected": e2e_exact_detected / max(det_found, 1),
        f"e2e_exact_iou{args.iou_threshold:.2f}": e2e_exact_loc_ok / max(total, 1),
        "detector_weights": args.detector_weights.resolve().as_posix(),
        "recognizer_weights": args.recognizer_weights.resolve().as_posix(),
        "test_index": args.test_index.resolve().as_posix(),
    }
    save_json(args.output, metrics)
    print(f"[eval] metrics written to {args.output.resolve().as_posix()}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
