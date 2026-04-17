from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SRCNNPatchDataset, build_frame_pairs, default_split_paths


class SRCNN(nn.Module):
    """Classic SRCNN: 9x9-64, 1x1-32, 5x5-1."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRCNN baseline")
    parser.add_argument(
        "--project1-root",
        type=Path,
        default=Path("/home/schung760/shared_data/project1"),
        help="Root containing train/ and val/ folders",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=33)
    parser.add_argument("--label-size", type=int, default=21)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--max-train-patches", type=int, default=None)
    parser.add_argument("--max-val-patches", type=int, default=None)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/checkpoints"),
    )
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0

    for lr_patch, hr_patch in loader:
        lr_patch = lr_patch.to(device, non_blocking=True)
        hr_patch = hr_patch.to(device, non_blocking=True)

        pred = model(lr_patch)
        loss = criterion(pred, hr_patch)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total += loss.item() * lr_patch.size(0)

    return total / len(loader.dataset)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_paths = default_split_paths(args.project1_root)
    train_pairs = build_frame_pairs(
        split_paths["train"].hr_root,
        split_paths["train"].lr_bicubic_root,
        max_pairs=args.max_train_pairs,
    )
    val_pairs = build_frame_pairs(
        split_paths["val"].hr_root,
        split_paths["val"].lr_bicubic_root,
        max_pairs=args.max_val_pairs,
    )

    print(f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}")

    train_set = SRCNNPatchDataset(
        train_pairs,
        patch_size=args.patch_size,
        label_size=args.label_size,
        stride=args.stride,
        max_patches=args.max_train_patches,
    )
    val_set = SRCNNPatchDataset(
        val_pairs,
        patch_size=args.patch_size,
        label_size=args.label_size,
        stride=args.stride,
        max_patches=args.max_val_patches,
    )

    print(f"train_patches={len(train_set)} val_patches={len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        torch.save(
            {"epoch": epoch, "model": model.state_dict()},
            args.save_dir / f"srcnn_epoch_{epoch:03d}.pt",
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                args.save_dir / "srcnn_best.pt",
            )
            print(f"  New best checkpoint saved to: {args.save_dir / 'srcnn_best.pt'}")


if __name__ == "__main__":
    main()
