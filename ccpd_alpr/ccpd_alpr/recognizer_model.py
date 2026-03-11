from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(in_ch: int, out_ch: int, k: int, s: int, p: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CRNNRecognizer(nn.Module):
    """
    A strong CRNN baseline for license plate text recognition.

    Input shape: [B, 3, 64, 256]
    Output shape: [T, B, num_classes]
    """

    def __init__(self, num_classes: int, hidden_size: int = 320, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(3, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 32 x 128
            _conv_bn_relu(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),  # 16 x 64
            _conv_bn_relu(128, 256, 3, 1, 1),
            _conv_bn_relu(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # 8 x 65
            _conv_bn_relu(256, 512, 3, 1, 1),
            _conv_bn_relu(512, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # 4 x 66
            _conv_bn_relu(512, 512, 2, 1, 0),  # 3 x 65
            _conv_bn_relu(512, 512, 2, 1, 0),  # 2 x 64
            nn.AdaptiveAvgPool2d((1, None)),  # 1 x W
        )
        self.sequence = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)  # [B, C, 1, W]
        feat = feat.squeeze(2).permute(2, 0, 1)  # [W, B, C]
        seq_out, _ = self.sequence(feat)
        logits = self.classifier(seq_out)  # [W, B, num_classes]
        return logits

