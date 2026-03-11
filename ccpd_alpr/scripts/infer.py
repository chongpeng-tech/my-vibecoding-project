from __future__ import annotations

import argparse
from pathlib import Path

from ccpd_alpr.service import ALPRService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end ALPR inference.")
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--recognizer-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/infer"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=1)
    parser.add_argument("--province-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--province-refine-threshold", type=float, default=0.26)
    parser.add_argument("--province-classifier-weights", type=Path, default=Path("runs/province_classifier/best.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ALPRService(
        detector_weights=args.detector_weights,
        recognizer_weights=args.recognizer_weights,
        device=args.device,
        province_refine=args.province_refine,
        province_refine_threshold=args.province_refine_threshold,
        province_classifier_weights=args.province_classifier_weights,
    )
    summary = service.infer_directory(
        source_dir=args.source,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    print(f"[infer] images={summary['num_images']} detected={summary['num_detected']}")
    print(f"[infer] visualizations: {summary['vis_dir']}")
    print(f"[infer] predictions jsonl: {summary['jsonl_path']}")
    print(f"[infer] predictions csv: {summary['csv_path']}")


if __name__ == "__main__":
    main()
