from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class PairSpec:
    sequence: str
    template_index: int
    search_index: int


def _crop(image: np.ndarray, box: np.ndarray, factor: float, size: int) -> np.ndarray:
    x, y, w, h = (float(v) for v in box)
    crop_size = max(1, math.ceil(math.sqrt(w * h) * factor))
    x1 = round(x + 0.5 * w - 0.5 * crop_size)
    y1 = round(y + 0.5 * h - 0.5 * crop_size)
    x2, y2 = x1 + crop_size, y1 + crop_size
    left, top = max(0, -x1), max(0, -y1)
    right = max(0, x2 - image.shape[1])
    bottom = max(0, y2 - image.shape[0])
    patch = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
    patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)


class GOT10kPairs:
    """Deterministic GOT-10k template/search sampler using one train root."""

    def __init__(
        self,
        root: Path,
        template_factor: float,
        template_size: int,
        search_factor: float,
        search_size: int,
        mean: list[float],
        std: list[float],
        seed: int,
        calibration_sequences: int,
        validation_sequences: int,
        max_frame_gap: int,
        min_visible_ratio: float,
        manifest_path: Path,
    ):
        self.root = root.resolve()
        if not (self.root / "list.txt").is_file():
            raise FileNotFoundError(f"GOT-10k list.txt not found under {self.root}")
        self.template_factor, self.template_size = template_factor, template_size
        self.search_factor, self.search_size = search_factor, search_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.max_frame_gap = max_frame_gap
        self.min_visible_ratio = min_visible_ratio
        listed = [x.strip() for x in (self.root / "list.txt").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        # Directory enumeration is much faster than probing every missing path
        # when only part of the official train archives has been extracted.
        extracted = {path.name for path in self.root.iterdir() if path.is_dir()}
        names = [name for name in listed if name in extracted]
        if not names:
            raise RuntimeError(f"No complete GOT-10k sequences found under {self.root}")
        if len(names) != len(listed):
            print(f"GOT-10k: using {len(names)}/{len(listed)} complete sequences")
        shuffled = names.copy()
        random.Random(seed).shuffle(shuffled)
        needed = calibration_sequences + validation_sequences
        if len(shuffled) < needed:
            raise ValueError(f"Need {needed} sequences but dataset only has {len(shuffled)}")
        self.splits = {
            "calibration": shuffled[:calibration_sequences],
            "validation": shuffled[calibration_sequences:needed],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"root": str(self.root), "seed": seed, **self.splits}, indent=2), encoding="utf-8")

    @staticmethod
    def _read_vector(path: Path, dtype: type) -> np.ndarray:
        return np.asarray([dtype(row[0]) for row in csv.reader(path.open("r", newline="", encoding="utf-8-sig"))])

    def specs(self, split: str, count: int, seed: int) -> list[PairSpec]:
        rng = random.Random(seed)
        result: list[PairSpec] = []
        attempts = 0
        names = self.splits[split]
        while len(result) < count and attempts < count * 50:
            attempts += 1
            name = rng.choice(names)
            seq = self.root / name
            if not all((seq / file).is_file() for file in ("groundtruth.txt", "absence.label", "cover.label", "00000001.jpg")):
                continue
            boxes = np.loadtxt(seq / "groundtruth.txt", delimiter=",", dtype=np.float32).reshape(-1, 4)
            absent = self._read_vector(seq / "absence.label", int)
            cover = self._read_vector(seq / "cover.label", float) / 8.0
            visible = np.flatnonzero((absent == 0) & (cover >= self.min_visible_ratio) & (boxes[:, 2] > 1) & (boxes[:, 3] > 1))
            if visible.size < 2:
                continue
            t = int(rng.choice(visible.tolist()))
            nearby = visible[(np.abs(visible - t) <= self.max_frame_gap) & (visible != t)]
            if nearby.size == 0:
                continue
            s = int(rng.choice(nearby.tolist()))
            result.append(PairSpec(name, t, s))
        if len(result) != count:
            raise RuntimeError(f"Could only sample {len(result)}/{count} valid GOT-10k pairs")
        return result

    def _normalize(self, rgb: np.ndarray) -> np.ndarray:
        value = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(value.transpose(2, 0, 1)[None])

    def load(self, spec: PairSpec) -> tuple[np.ndarray, np.ndarray]:
        seq = self.root / spec.sequence
        boxes = np.loadtxt(seq / "groundtruth.txt", delimiter=",", dtype=np.float32).reshape(-1, 4)
        template = cv2.imread(str(seq / f"{spec.template_index + 1:08d}.jpg"), cv2.IMREAD_COLOR)
        search = cv2.imread(str(seq / f"{spec.search_index + 1:08d}.jpg"), cv2.IMREAD_COLOR)
        if template is None or search is None:
            raise RuntimeError(f"Failed to read frames from {seq}")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        search = cv2.cvtColor(search, cv2.COLOR_BGR2RGB)
        return (
            self._normalize(_crop(template, boxes[spec.template_index], self.template_factor, self.template_size)),
            self._normalize(_crop(search, boxes[spec.search_index], self.search_factor, self.search_size)),
        )

    def load_raw(self, spec: PairSpec) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
        """Return BGR frames and xywh boxes for tracker-level accuracy evaluation."""
        seq = self.root / spec.sequence
        boxes = np.loadtxt(seq / "groundtruth.txt", delimiter=",", dtype=np.float32).reshape(-1, 4)
        template = cv2.imread(str(seq / f"{spec.template_index + 1:08d}.jpg"), cv2.IMREAD_COLOR)
        search = cv2.imread(str(seq / f"{spec.search_index + 1:08d}.jpg"), cv2.IMREAD_COLOR)
        if template is None or search is None:
            raise RuntimeError(f"Failed to read frames from {seq}")
        return template, search, boxes[spec.template_index].tolist(), boxes[spec.search_index].tolist()

    def iter_arrays(self, specs: list[PairSpec]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for spec in specs:
            yield self.load(spec)
