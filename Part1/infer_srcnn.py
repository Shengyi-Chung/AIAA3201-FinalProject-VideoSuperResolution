from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import build_frame_pairs, default_split_paths, load_bgr, load_y_channel
from train_srcnn import SRCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SRCNN inference and save visual previews")
    parser.add_argument("--project1-root", type=Path, default=Path("/home/schung760/shared_data/project1"))
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument(
        "--srcnn-ckpt",
        type=Path,
        default=Path("/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/checkpoints/srcnn_best.pt"),
    )
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/outputs_srcnn_preview"),
    )
    return parser.parse_args()


def y_to_bgr(y: np.ndarray) -> np.ndarray:
    y_u8 = (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(y_u8, cv2.COLOR_GRAY2BGR)


def center_crop(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    img_h, img_w = img.shape[:2]
    top = (img_h - target_h) // 2
    left = (img_w - target_w) // 2
    return img[top : top + target_h, left : left + target_w]


def main() -> None:
    args = parse_args()

    if not args.srcnn_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.srcnn_ckpt}")

    split_paths = default_split_paths(args.project1_root)[args.split]
    pairs = build_frame_pairs(
        split_paths.hr_root,
        split_paths.lr_bicubic_root,
        max_pairs=args.max_pairs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    state = torch.load(args.srcnn_ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, (lr_path, hr_path) in enumerate(pairs):
            inp_y = load_y_channel(lr_path)
            gt_y = load_y_channel(hr_path)

            if inp_y.shape != gt_y.shape:
                gt_h, gt_w = gt_y.shape
                inp_y = cv2.resize(inp_y, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)

            x = torch.from_numpy(inp_y).unsqueeze(0).unsqueeze(0).to(device)
            pred_y = model(x).squeeze(0).squeeze(0).cpu().numpy()
            pred_y = np.clip(pred_y, 0.0, 1.0)

            if pred_y.shape != gt_y.shape:
                gt_y = center_crop(gt_y, pred_y.shape)

            if inp_y.shape != pred_y.shape:
                inp_y = center_crop(inp_y, pred_y.shape)

            inp_img = y_to_bgr(inp_y)
            pred_img = y_to_bgr(pred_y)
            gt_img = y_to_bgr(gt_y)

            panel = np.concatenate([inp_img, pred_img, gt_img], axis=1)

            stem = f"{idx:03d}_{hr_path.parent.name}_{hr_path.stem}"
            cv2.imwrite(str(args.out_dir / f"{stem}_input.png"), inp_img)
            cv2.imwrite(str(args.out_dir / f"{stem}_srcnn.png"), pred_img)
            cv2.imwrite(str(args.out_dir / f"{stem}_gt.png"), gt_img)
            cv2.imwrite(str(args.out_dir / f"{stem}_panel.png"), panel)

    print(f"Saved {len(pairs)} preview samples to: {args.out_dir}")


if __name__ == "__main__":
    main()
