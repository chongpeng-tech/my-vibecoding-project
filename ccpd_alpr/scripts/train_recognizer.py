from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccpd_alpr.constants import default_charset
from ccpd_alpr.ocr_dataset import CCPDOCRDataset, ctc_collate
from ccpd_alpr.recognizer_model import CRNNRecognizer
from ccpd_alpr.tokenizer import CTCLabelConverter
from ccpd_alpr.utils import ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRNN recognizer on CCPD.")
    parser.add_argument("--train-index", type=Path, default=Path("data/index/train.jsonl"))
    parser.add_argument("--val-index", type=Path, default=Path("data/index/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/recognizer/crnn_ctc"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_texts_from_jsonl(index_file: Path) -> list[str]:
    texts: list[str] = []
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            texts.append(data["plate_text"])
    return texts


def build_charset(train_index: Path) -> list[str]:
    chars = set(default_charset())
    for text in load_texts_from_jsonl(train_index):
        for ch in text:
            chars.add(ch)
    return sorted(chars)


@torch.inference_mode()
def evaluate(
    model: CRNNRecognizer,
    dataloader: DataLoader,
    converter: CTCLabelConverter,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    exact_correct = 0
    char_correct = 0
    char_total = 0
    for images, texts in tqdm(dataloader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        log_probs = logits.log_softmax(dim=2)
        targets, target_lengths = converter.encode_batch(texts)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = converter.decode_batch(log_probs)
        for pred, gt in zip(preds, texts):
            if pred == gt:
                exact_correct += 1
            n = max(len(gt), 1)
            char_total += n
            char_correct += sum(1 for a, b in zip(pred, gt) if a == b)

    val_loss = total_loss / max(total_samples, 1)
    exact_acc = exact_correct / max(total_samples, 1)
    char_acc = char_correct / max(char_total, 1)
    return val_loss, exact_acc, char_acc


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    charset = build_charset(args.train_index)
    converter = CTCLabelConverter(charset=charset)
    (output_dir / "charset.txt").write_text("\n".join(charset), encoding="utf-8")

    train_ds = CCPDOCRDataset(
        index_file=args.train_index,
        image_width=args.width,
        image_height=args.height,
        augment=True,
    )
    val_ds = CCPDOCRDataset(
        index_file=args.val_index,
        image_width=args.width,
        image_height=args.height,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=ctc_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=ctc_collate,
    )

    model = CRNNRecognizer(num_classes=converter.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    start_epoch = 0
    best_acc = -1.0
    if args.resume is not None and args.resume.exists() and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_acc = float(ckpt.get("best_acc", -1.0))
        print(f"[train] resumed from {args.resume}, start epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0
        seen = 0
        for images, texts in pbar:
            images = images.to(device, non_blocking=True)
            targets, target_lengths = converter.encode_batch(texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                log_probs = logits.log_softmax(dim=2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=log_probs.size(0),
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            pbar.set_postfix(loss=running_loss / max(seen, 1), lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        val_loss, exact_acc, char_acc = evaluate(model, val_loader, converter, criterion, device)
        print(
            f"[epoch {epoch + 1}] train_loss={running_loss / max(seen, 1):.5f} "
            f"val_loss={val_loss:.5f} val_exact={exact_acc:.5f} val_char={char_acc:.5f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "charset": charset,
            "best_acc": best_acc,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if exact_acc > best_acc:
            best_acc = exact_acc
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"[train] new best exact acc: {best_acc:.5f}")

    print(f"[train] done, best val exact acc = {best_acc:.5f}")


if __name__ == "__main__":
    main()
