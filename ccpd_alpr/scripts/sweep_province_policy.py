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

ANHUI_CHAR = "\u7696"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep province replacement policies.")
    parser.add_argument("--index-file", type=Path, default=Path("data/index/val.jsonl"))
    parser.add_argument("--recognizer-weights", type=Path, required=True)
    parser.add_argument("--province-classifier-weights", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.26)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def load_samples(index_file: Path, max_samples: int = 0) -> list[dict]:
    rows: list[dict] = []
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def apply_policy(
    policy: str,
    base_text: str,
    ocr_conf: float,
    candidate: str,
    cand_conf: float,
    margin: float,
    threshold: float,
    province_chars: set[str],
) -> str:
    if not base_text:
        return base_text
    first = base_text[0]
    if first not in province_chars:
        if cand_conf >= threshold:
            return candidate + base_text[1:]
        return base_text

    if policy == "current":
        if first == ANHUI_CHAR and candidate != ANHUI_CHAR:
            if cand_conf >= threshold + 0.08 and margin >= 0.03:
                return candidate + base_text[1:]
            return base_text
        if candidate != first and cand_conf >= threshold + 0.18 and margin >= 0.08 and ocr_conf <= 0.95:
            return candidate + base_text[1:]
        return base_text

    if policy == "moderate":
        if candidate != first and cand_conf >= threshold + 0.08 and margin >= 0.04:
            return candidate + base_text[1:]
        return base_text

    if policy == "aggressive":
        if candidate != first and cand_conf >= threshold and margin >= 0.03:
            return candidate + base_text[1:]
        return base_text

    if policy == "high_conf":
        if candidate != first and cand_conf >= max(threshold, 0.55):
            return candidate + base_text[1:]
        return base_text

    return base_text


def evaluate_policy(pred_rows: list[dict], policy: str, threshold: float, province_chars: set[str]) -> dict[str, float]:
    total = 0
    first_ok = 0
    full_ok = 0
    non_anhui_total = 0
    non_anhui_first_ok = 0
    replace_count = 0
    for row in pred_rows:
        gt_text = row["gt_text"]
        base_text = row["base_text"]
        out_text = apply_policy(
            policy=policy,
            base_text=base_text,
            ocr_conf=row["ocr_conf"],
            candidate=row["candidate"],
            cand_conf=row["cand_conf"],
            margin=row["margin"],
            threshold=threshold,
            province_chars=province_chars,
        )
        if out_text != base_text:
            replace_count += 1
        total += 1
        if out_text[:1] == gt_text[:1]:
            first_ok += 1
        if out_text == gt_text:
            full_ok += 1
        if gt_text[:1] != ANHUI_CHAR:
            non_anhui_total += 1
            if out_text[:1] == gt_text[:1]:
                non_anhui_first_ok += 1
    return {
        "policy": policy,
        "samples": total,
        "replace_count": replace_count,
        "first_char_acc": first_ok / max(total, 1),
        "full_plate_acc": full_ok / max(total, 1),
        "non_anhui_first_char_acc": non_anhui_first_ok / max(non_anhui_total, 1),
        "non_anhui_samples": non_anhui_total,
    }


def main() -> None:
    args = parse_args()
    rows = load_samples(args.index_file, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("No samples loaded.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    recognizer, converter = load_recognizer(args.recognizer_weights, device=device)
    province_clf = ProvinceClassifier(args.province_classifier_weights, device=device)
    province_chars = set(province_charset())

    pred_rows: list[dict] = []
    for row in tqdm(rows, desc="collect"):
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
        pred_rows.append(
            {
                "gt_text": gt_text,
                "base_text": base_text,
                "ocr_conf": float(ocr_conf),
                "candidate": pred.char,
                "cand_conf": float(pred.conf),
                "margin": margin,
            }
        )

    policies = ["current", "moderate", "aggressive", "high_conf"]
    results = [evaluate_policy(pred_rows, p, threshold=float(args.threshold), province_chars=province_chars) for p in policies]
    print(json.dumps({"threshold": float(args.threshold), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
