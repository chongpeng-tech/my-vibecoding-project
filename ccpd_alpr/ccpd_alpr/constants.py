from __future__ import annotations

from typing import Iterable

# Official CCPD mappings.
PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "警",
    "学",
    "O",
]

ALPHABETS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
]

ADS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "O",
]

# A larger charset for better out-of-set generalization.
EXTRA_PLATE_CHARS = [
    "挂",
    "使",
    "领",
    "港",
    "澳",
    "电",
    "应",
    "急",
    "民",
    "航",
]


def safe_pick(table: list[str], index: int) -> str:
    if 0 <= index < len(table):
        return table[index]
    return "O"


def decode_plate_indices(indices: Iterable[int]) -> str:
    idx = list(indices)
    if len(idx) < 2:
        return ""
    chars = [safe_pick(PROVINCES, idx[0]), safe_pick(ALPHABETS, idx[1])]
    chars.extend(safe_pick(ADS, token) for token in idx[2:])
    return "".join(chars)


def default_charset() -> list[str]:
    chars = set(PROVINCES) | set(ALPHABETS) | set(ADS) | set(EXTRA_PLATE_CHARS)
    chars.discard("O")
    chars.add("O")
    return sorted(chars)


def province_charset() -> list[str]:
    # "O" is kept in original list for compatibility but not a real province mark.
    return [p for p in PROVINCES if p != "O"]
