from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ccpd_alpr.constants import province_charset
from ccpd_alpr.geometry import warp_plate
from ccpd_alpr.pipeline import load_recognizer, recognize_plate
from ccpd_alpr.province_classifier import ProvinceClassifier
from ccpd_alpr.utils import ensure_dir, save_json

ANHUI_CHAR = "\u7696"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate first-char refinement on CCPD index.")
    parser.add_argument("--index-file", type=Path, default=Path("data/index/val.jsonl"))
    parser.add_argument("--recognizer-weights", type=Path, required=True)
    parser.add_argument("--province-classifier-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/eval/province_refine_metrics.json"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.26)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--non-anhui-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_samples(index_file: Path) -> list[dict]:
    rows: list[dict] = []
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def refine_first_char(
    plate_text: str,
    ocr_conf: float,
    candidate: str,
    cand_conf: float,
    cand_margin: float,
    threshold: float,
    province_chars: set[str],
) -> tuple[str, bool]:
    if not plate_text:
        return plate_text, False
    first = plate_text[0]
    _ = ocr_conf
    if cand_conf < threshold:
        return plate_text, False
    if first not in province_chars:
        return candidate + plate_text[1:], True
    if candidate != first and cand_conf >= threshold + 0.08 and cand_margin >= 0.04:
        return candidate + plate_text[1:], True
    return plate_text, False


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output.parent)
    _ = output_dir

    rows = load_samples(args.index_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if args.non_anhui_only:
        rows = [r for r in rows if r.get("plate_text", "")[:1] != ANHUI_CHAR]
    if not rows:
        raise RuntimeError("No samples to evaluate.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    recognizer, converter = load_recognizer(args.recognizer_weights, device=device)
    province_clf = ProvinceClassifier(args.province_classifier_weights, device=device)
    province_chars = set(province_charset())

    total = 0
    first_ok_base = 0
    first_ok_refine = 0
    full_ok_base = 0
    full_ok_refine = 0
    non_anhui_total = 0
    non_anhui_first_ok_base = 0
    non_anhui_first_ok_refine = 0

    for row in tqdm(rows, desc="eval-province-refine"):
        image = cv2.imread(row["image_path"])
        if image is None:
            continue
        corners = np.asarray(row["corners"], dtype=np.float32)
        gt_text = row.get("plate_text", "")
        if not gt_text:
            continue

        patch = warp_plate(image, corners, width=256, height=64)
        base_text, ocr_conf = recognize_plate(recognizer, converter, patch, device=device)
        pred = province_clf.predict(patch)
        margin = float(pred.top3[0][1] - pred.top3[1][1]) if len(pred.top3) > 1 else float(pred.conf)
        refine_text, _ = refine_first_char(
            plate_text=base_text,
            ocr_conf=float(ocr_conf),
            candidate=pred.char,
            cand_conf=float(pred.conf),
            cand_margin=margin,
            threshold=float(args.threshold),
            province_chars=province_chars,
        )

        total += 1
        if base_text[:1] == gt_text[:1]:
            first_ok_base += 1
        if refine_text[:1] == gt_text[:1]:
            first_ok_refine += 1
        if base_text == gt_text:
            full_ok_base += 1
        if refine_text == gt_text:
            full_ok_refine += 1

        if gt_text[:1] != ANHUI_CHAR:
            non_anhui_total += 1
            if base_text[:1] == gt_text[:1]:
                non_anhui_first_ok_base += 1
            if refine_text[:1] == gt_text[:1]:
                non_anhui_first_ok_refine += 1

    metrics = {
        "num_samples": total,
        "first_char_acc_base": first_ok_base / max(total, 1),
        "first_char_acc_refine": first_ok_refine / max(total, 1),
        "full_plate_acc_base": full_ok_base / max(total, 1),
        "full_plate_acc_refine": full_ok_refine / max(total, 1),
        "non_anhui_samples": non_anhui_total,
        "non_anhui_first_char_acc_base": non_anhui_first_ok_base / max(non_anhui_total, 1),
        "non_anhui_first_char_acc_refine": non_anhui_first_ok_refine / max(non_anhui_total, 1),
        "threshold": float(args.threshold),
        "recognizer_weights": args.recognizer_weights.resolve().as_posix(),
        "province_classifier_weights": args.province_classifier_weights.resolve().as_posix(),
        "index_file": args.index_file.resolve().as_posix(),
    }
    save_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
