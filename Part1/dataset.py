from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitPaths:
    hr_root: Path
    lr_bicubic_root: Path


def default_split_paths(project1_root: str | Path) -> dict[str, SplitPaths]:
    """Paths for your existing dataset under shared_data/project1."""
    root = Path(project1_root)
    return {
        "train": SplitPaths(
            hr_root=root / "train" / "train_sharp",
            lr_bicubic_root=root / "train" / "train_sharp_bicubic" / "X4",
        ),
        "val": SplitPaths(
            hr_root=root / "val" / "val_sharp",
            lr_bicubic_root=root / "val" / "val_sharp_bicubic" / "X4",
        ),
    }


def _sorted_pngs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.png"))


def build_frame_pairs(
    hr_root: str | Path,
    lr_bicubic_root: str | Path,
    max_pairs: int | None = None,
) -> List[Tuple[Path, Path]]:
    """Build aligned (lr_bicubic_up, hr) pairs by sequence folder + frame name."""
    hr_root = Path(hr_root)
    lr_root = Path(lr_bicubic_root)

    if not hr_root.exists() or not lr_root.exists():
        raise FileNotFoundError(f"Path not found: HR={hr_root}, LR={lr_root}")

    pairs: List[Tuple[Path, Path]] = []
    for seq_dir in sorted(p for p in hr_root.iterdir() if p.is_dir()):
        lr_seq = lr_root / seq_dir.name
        if not lr_seq.exists():
            continue

        lr_map = {p.name: p for p in _sorted_pngs(lr_seq)}
        for hr in _sorted_pngs(seq_dir):
            lr = lr_map.get(hr.name)
            if lr is not None:
                pairs.append((lr, hr))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs

    if not pairs:
        raise RuntimeError(
            f"No matched frame pairs found under HR={hr_root} and LR={lr_root}."
        )
    return pairs


def load_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def bgr_to_y_float01(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float32) / 255.0


def load_y_channel(path: str | Path) -> np.ndarray:
    return bgr_to_y_float01(load_bgr(path))


def align_lr_to_hr(lr_y: np.ndarray, hr_y: np.ndarray) -> np.ndarray:
    """Upsample LR Y channel to HR size when shapes do not match."""
    if lr_y.shape == hr_y.shape:
        return lr_y

    hr_h, hr_w = hr_y.shape
    return cv2.resize(lr_y, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)


class SRCNNPatchDataset(Dataset):
    """Patch-level dataset on Y channel for SRCNN baseline training."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        patch_size: int = 33,
        stride: int = 14,
        max_patches: int | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.cache: dict[Tuple[Path, Path], Tuple[np.ndarray, np.ndarray]] = {}
        self.samples: List[Tuple[Path, Path, int, int]] = []

        for lr_path, hr_path in pairs:
            y_lr = load_y_channel(lr_path)
            y_hr = load_y_channel(hr_path)

            y_lr = align_lr_to_hr(y_lr, y_hr)
            self.cache[(lr_path, hr_path)] = (y_lr, y_hr)

            h, w = y_lr.shape
            if h < patch_size or w < patch_size:
                continue

            for top in range(0, h - patch_size + 1, stride):
                for left in range(0, w - patch_size + 1, stride):
                    self.samples.append((lr_path, hr_path, top, left))
                    if max_patches is not None and len(self.samples) >= max_patches:
                        break
                if max_patches is not None and len(self.samples) >= max_patches:
                    break

            if max_patches is not None and len(self.samples) >= max_patches:
                break

        if not self.samples:
            raise RuntimeError("No patches generated. Check patch_size/stride and paths.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_path, hr_path, top, left = self.samples[idx]
        y_lr, y_hr = self.cache[(lr_path, hr_path)]

        lr_patch = y_lr[top : top + self.patch_size, left : left + self.patch_size]
        hr_patch = y_hr[top : top + self.patch_size, left : left + self.patch_size]

        return torch.from_numpy(lr_patch).unsqueeze(0), torch.from_numpy(hr_patch).unsqueeze(0)
