from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import cv2
import yaml

from ccpd_alpr.geometry import bbox_xyxy_to_yolo
from ccpd_alpr.parser import CCPDRecord, parse_ccpd_filename
from ccpd_alpr.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CCPD for detector + OCR training.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root dir of CCPD images.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Output root for prepared files.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="For debugging only. 0 means using all samples.",
    )
    return parser.parse_args()


def list_images(dataset_root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    return [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix in exts]


def parse_records(image_paths: list[Path]) -> list[CCPDRecord]:
    records: list[CCPDRecord] = []
    for path in image_paths:
        try:
            records.append(parse_ccpd_filename(path))
        except ValueError:
            continue
    return records


def split_records(
    records: list[CCPDRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[CCPDRecord], list[CCPDRecord], list[CCPDRecord]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def ensure_images_link(images_all_dir: Path, dataset_root: Path) -> None:
    if images_all_dir.exists():
        return
    images_all_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(dataset_root), str(images_all_dir), target_is_directory=True)
        return
    except Exception:
        pass
    if os.name == "nt":
        # Junction fallback on Windows.
        cmd = ["cmd", "/c", "mklink", "/J", str(images_all_dir), str(dataset_root)]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            return
    raise RuntimeError(f"Failed to create link from {images_all_dir} to {dataset_root}")


def infer_image_size(records: list[CCPDRecord]) -> tuple[int, int]:
    for rec in records[:64]:
        img = cv2.imread(str(rec.image_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    raise RuntimeError("Cannot read any image to infer image size.")


def build_label_line(record: CCPDRecord, img_w: int, img_h: int) -> str:
    cx, cy, bw, bh = bbox_xyxy_to_yolo(record.bbox_xyxy, img_w, img_h)
    values = [0, cx, cy, bw, bh]
    for point in record.corners_rd_ld_lu_ru:
        x = float(point[0]) / img_w
        y = float(point[1]) / img_h
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        values.extend([x, y, 2])
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)


def write_detector_files(
    records: list[CCPDRecord],
    split_name: str,
    detector_root: Path,
    images_all_dir: Path,
    img_w: int,
    img_h: int,
) -> None:
    labels_all = ensure_dir(detector_root / "labels" / "all")
    split_file = ensure_dir(detector_root / "splits") / f"{split_name}.txt"

    with split_file.open("w", encoding="utf-8") as split_writer:
        for record in records:
            label_path = labels_all / f"{record.image_path.stem}.txt"
            label_path.write_text(build_label_line(record, img_w, img_h) + "\n", encoding="utf-8")
            # Keep the `/images/` segment for Ultralytics label-path mapping.
            image_path = (images_all_dir / record.image_path.name).absolute()
            split_writer.write(image_path.as_posix() + "\n")


def write_index_jsonl(records: list[CCPDRecord], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            row = {
                "image_path": rec.image_path.resolve().as_posix(),
                "file_name": rec.image_path.name,
                "bbox": list(rec.bbox_xyxy),
                "corners": rec.corners_rd_ld_lu_ru.tolist(),
                "plate_indices": rec.plate_indices,
                "plate_text": rec.plate_text,
                "brightness": rec.brightness,
                "blurriness": rec.blurriness,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_yaml(detector_root: Path) -> None:
    yaml_path = detector_root / "ccpd_pose.yaml"
    payload = {
        "path": detector_root.resolve().as_posix(),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "test": "splits/test.txt",
        "names": {0: "plate"},
        "kpt_shape": [4, 3],
        "flip_idx": [0, 1, 2, 3],
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    detector_root = output_root / "detector"
    index_root = output_root / "index"

    print(f"[prepare] scanning images in {dataset_root}")
    image_paths = list_images(dataset_root)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]
    print(f"[prepare] found {len(image_paths)} images")

    print("[prepare] parsing annotations from filenames")
    records = parse_records(image_paths)
    if not records:
        raise RuntimeError("No valid CCPD samples were parsed.")
    print(f"[prepare] parsed {len(records)} valid samples")

    train, val, test = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    print(f"[prepare] split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    print("[prepare] inferring image shape")
    img_w, img_h = infer_image_size(records)
    print(f"[prepare] image size inferred as width={img_w}, height={img_h}")

    print("[prepare] creating detector image link")
    images_all_dir = detector_root / "images" / "all"
    ensure_images_link(images_all_dir, dataset_root)

    print("[prepare] writing detector labels and split files")
    write_detector_files(train, "train", detector_root, images_all_dir, img_w, img_h)
    write_detector_files(val, "val", detector_root, images_all_dir, img_w, img_h)
    write_detector_files(test, "test", detector_root, images_all_dir, img_w, img_h)
    write_dataset_yaml(detector_root)

    print("[prepare] writing OCR index files")
    write_index_jsonl(train, index_root / "train.jsonl")
    write_index_jsonl(val, index_root / "val.jsonl")
    write_index_jsonl(test, index_root / "test.jsonl")

    save_json(
        output_root / "meta.json",
        {
            "dataset_root": dataset_root.as_posix(),
            "output_root": output_root.as_posix(),
            "num_samples": len(records),
            "image_size": [img_w, img_h],
            "splits": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "seed": args.seed,
        },
    )
    print("[prepare] done")


if __name__ == "__main__":
    main()
