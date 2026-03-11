from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class CTCLabelConverter:
    charset: list[str]
    blank_idx: int = field(init=False, default=0)
    char_to_idx: dict[str, int] = field(init=False)
    idx_to_char: dict[int, str] = field(init=False)
    num_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.charset)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.charset)}
        self.num_classes = len(self.charset) + 1

    def encode_text(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded: list[int] = []
        lengths: list[int] = []
        for text in texts:
            token_ids = self.encode_text(text)
            encoded.extend(token_ids)
            lengths.append(len(token_ids))
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    def decode_batch(self, log_probs: torch.Tensor) -> list[str]:
        # log_probs: [T, B, C]
        probs = log_probs.detach().argmax(dim=2).cpu().numpy().T
        decoded_texts: list[str] = []
        for row in probs:
            chars: list[str] = []
            prev = self.blank_idx
            for token in row:
                if token != self.blank_idx and token != prev:
                    chars.append(self.idx_to_char.get(int(token), ""))
                prev = int(token)
            decoded_texts.append("".join(chars))
        return decoded_texts
