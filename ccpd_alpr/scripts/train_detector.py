from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO pose detector for CCPD.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/detector/ccpd_pose.yaml"),
        help="Path to YOLO dataset yaml.",
    )
    parser.add_argument("--model", type=str, default="yolov8x-pose.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, default=Path("detector"))
    parser.add_argument("--name", type=str, default="yolov8x_pose_ccpd")
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--plots", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data.as_posix(),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project.as_posix(),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        optimizer="AdamW",
        cos_lr=True,
        close_mosaic=0,
        hsv_h=0.015,
        hsv_s=0.65,
        hsv_v=0.45,
        degrees=10.0,
        translate=0.12,
        scale=0.5,
        shear=2.0,
        perspective=0.001,
        fliplr=0.0,
        flipud=0.0,
        erasing=0.15,
        cache="disk",
        rect=False,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
