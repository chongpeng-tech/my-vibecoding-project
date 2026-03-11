from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from ccpd_alpr.constants import province_charset
from ccpd_alpr.geometry import warp_plate
from ccpd_alpr.province_classifier import ProvinceClassifierNet, crop_first_char_region
from ccpd_alpr.utils import ensure_dir, seed_everything

ANHUI_CHAR = "\u7696"


def _augment_plate(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    if random.random() < 0.6:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-25.0, 25.0)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if random.random() < 0.4:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0)
    if random.random() < 0.35:
        noise = np.random.normal(0.0, random.uniform(2.0, 9.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _augment_first_char(gray: np.ndarray) -> np.ndarray:
    out = gray.copy()
    if random.random() < 0.6:
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-3.0, 3.0)
        ty = random.uniform(-3.0, 3.0)
        mat = cv2.getRotationMatrix2D((32.0, 32.0), angle, scale)
        mat[:, 2] += (tx, ty)
        out = cv2.warpAffine(out, mat, (64, 64), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=0)
    if random.random() < 0.25:
        out = cv2.erode(out, np.ones((2, 2), dtype=np.uint8), iterations=1)
    if random.random() < 0.25:
        out = cv2.dilate(out, np.ones((2, 2), dtype=np.uint8), iterations=1)
    if random.random() < 0.2:
        x1 = random.randint(0, 50)
        y1 = random.randint(0, 50)
        x2 = min(63, x1 + random.randint(6, 16))
        y2 = min(63, y1 + random.randint(6, 16))
        fill = random.randint(0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), int(fill), thickness=-1)
    return out


class CCPDProvinceDataset(Dataset):
    def __init__(
        self,
        index_file: str | Path,
        chars: list[str],
        plate_width: int = 256,
        plate_height: int = 64,
        char_size: int = 64,
        augment: bool = False,
        max_samples: int = 0,
    ) -> None:
        self.index_file = Path(index_file)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.plate_width = int(plate_width)
        self.plate_height = int(plate_height)
        self.char_size = int(char_size)
        self.augment = augment
        self.samples = self._load_samples(max_samples=max_samples)
        if not self.samples:
            raise RuntimeError(f"No valid samples loaded from {self.index_file}")
        self.labels = [int(s["label"]) for s in self.samples]

    def _load_samples(self, max_samples: int = 0) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.index_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("plate_text", "")
                if not text:
                    continue
                first_char = text[0]
                if first_char not in self.char_to_idx:
                    continue
                records.append(
                    {
                        "image_path": row["image_path"],
                        "corners": np.asarray(row["corners"], dtype=np.float32),
                        "label": self.char_to_idx[first_char],
                    }
                )
                if max_samples > 0 and len(records) >= max_samples:
                    break
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def _read_sample(self, idx: int) -> tuple[np.ndarray, int]:
        rec = self.samples[idx]
        image = cv2.imread(rec["image_path"])
        if image is None:
            raise FileNotFoundError(rec["image_path"])
        plate = warp_plate(
            image_bgr=image,
            corners_rd_ld_lu_ru=rec["corners"],
            width=self.plate_width,
            height=self.plate_height,
        )
        if self.augment:
            plate = _augment_plate(plate)
        first_char = crop_first_char_region(plate, size=self.char_size)
        if self.augment:
            first_char = _augment_first_char(first_char)
        return first_char, int(rec["label"])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        trials = 0
        while trials < 8:
            current_idx = idx if trials == 0 else random.randint(0, len(self.samples) - 1)
            try:
                first_char, label = self._read_sample(current_idx)
                x = torch.from_numpy(first_char).unsqueeze(0).float() / 255.0
                y = torch.tensor(label, dtype=torch.long)
                return x, y
            except Exception:
                trials += 1
        raise RuntimeError("Failed to read valid sample after retries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train province first-char classifier with real CCPD crops.")
    parser.add_argument("--train-index", type=Path, default=Path("data/index/train.jsonl"))
    parser.add_argument("--val-index", type=Path, default=Path("data/index/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/province_classifier"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--balance-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance-power", type=float, default=0.5)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--plate-width", type=int, default=256)
    parser.add_argument("--plate-height", type=int, default=64)
    parser.add_argument("--char-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def build_sampler(labels: list[int], power: float) -> tuple[WeightedRandomSampler, Counter[int]]:
    counter: Counter[int] = Counter(labels)
    weights = [1.0 / float(max(counter[label], 1)) ** float(power) for label in labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
    return sampler, counter


@torch.inference_mode()
def evaluate(
    model: ProvinceClassifierNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    anhui_idx: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    non_anhui_total = 0
    non_anhui_correct = 0
    anhui_total = 0
    anhui_correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += int((pred == y).sum().item())

        mask_anhui = y == anhui_idx
        if int(mask_anhui.sum().item()) > 0:
            anhui_total += int(mask_anhui.sum().item())
            anhui_correct += int((pred[mask_anhui] == y[mask_anhui]).sum().item())
        mask_non_anhui = ~mask_anhui
        if int(mask_non_anhui.sum().item()) > 0:
            non_anhui_total += int(mask_non_anhui.sum().item())
            non_anhui_correct += int((pred[mask_non_anhui] == y[mask_non_anhui]).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "anhui_acc": anhui_correct / max(anhui_total, 1),
        "non_anhui_acc": non_anhui_correct / max(non_anhui_total, 1),
        "samples": float(total),
        "non_anhui_samples": float(non_anhui_total),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    chars = province_charset()
    anhui_idx = chars.index(ANHUI_CHAR)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = CCPDProvinceDataset(
        index_file=args.train_index,
        chars=chars,
        plate_width=args.plate_width,
        plate_height=args.plate_height,
        char_size=args.char_size,
        augment=True,
        max_samples=args.max_train_samples,
    )
    val_ds = CCPDProvinceDataset(
        index_file=args.val_index,
        chars=chars,
        plate_width=args.plate_width,
        plate_height=args.plate_height,
        char_size=args.char_size,
        augment=False,
        max_samples=args.max_val_samples,
    )

    train_counter = Counter(train_ds.labels)
    class_weights = []
    for i in range(len(chars)):
        class_weights.append(1.0 / float(max(train_counter.get(i, 1), 1)) ** float(args.class_weight_power))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    class_weights_t = class_weights_t / class_weights_t.mean()

    sampler = None
    if args.balance_sampler:
        sampler, _ = build_sampler(train_ds.labels, power=args.balance_power)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = ProvinceClassifierNet(len(chars)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t.to(device),
        label_smoothing=float(max(0.0, min(args.label_smoothing, 0.2))),
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        seen = 0
        running_loss = 0.0
        running_correct = 0
        pbar = tqdm(train_loader, desc=f"province epoch {epoch + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(dim=1)
            running_correct += int((pred == y).sum().item())
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(
                loss=running_loss / max(seen, 1),
                acc=running_correct / max(seen, 1),
                lr=optimizer.param_groups[0]["lr"],
            )

        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device, anhui_idx)
        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)
        score = 0.75 * float(val_metrics["non_anhui_acc"]) + 0.25 * float(val_metrics["acc"])
        print(
            f"[epoch {epoch + 1}] train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_acc={val_metrics['acc']:.5f} "
            f"val_non_anhui={val_metrics['non_anhui_acc']:.5f} score={score:.5f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "charset": chars,
            "best_score": best_score,
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if score > best_score:
            best_score = score
            ckpt["best_score"] = best_score
            torch.save(ckpt, output_dir / "best.pt")
            print(f"[train] new best score={best_score:.5f}")

    print(f"[train] done. best_score={best_score:.5f}")


if __name__ == "__main__":
    main()
