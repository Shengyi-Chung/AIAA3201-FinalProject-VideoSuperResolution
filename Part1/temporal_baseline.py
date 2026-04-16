from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr_y(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0.0:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_y(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(structural_similarity(pred, gt, data_range=1.0))


def bgr_to_y_float01(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float32) / 255.0


def temporal_weighted_average(frames: List[np.ndarray], center: int, weights: List[float]) -> np.ndarray:
    half = len(weights) // 2
    out = np.zeros_like(frames[0], dtype=np.float32)

    for i, wgt in enumerate(weights):
        idx = int(np.clip(center + (i - half), 0, len(frames) - 1))
        out += wgt * frames[idx]
    return np.clip(out, 0.0, 1.0)


def unsharp_mask(img: np.ndarray, sigma: float = 1.0, amount: float = 0.6) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = img + amount * (img - blur)
    return np.clip(sharp, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal baseline on Vimeo test set")
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("/home/schung760/shared_data/project1/vimeo_super_resolution_test"),
    )
    parser.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 0.25])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--amount", type=float, default=0.6)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("/home/schung760/AIAA3201-FinalProject-VideoSuperResolution/Part1/temporal_metrics.csv"),
    )
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("/home/schung760/AIAA3201-FinalProject-VideoSuperResolution/Part1/temporal_preview"),
    )
    return parser.parse_args()


def list_center_frames(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("im4.png"))


def main() -> None:
    args = parse_args()
    if len(args.weights) % 2 == 0:
        raise ValueError("weights length must be odd, e.g. 3 or 5")

    weights = np.array(args.weights, dtype=np.float32)
    weights = (weights / weights.sum()).tolist()

    input_root = args.test_root / "input"
    target_root = args.test_root / "target"
    centers = list_center_frames(input_root)

    if not centers:
        raise RuntimeError(f"No im4.png found under {input_root}")

    if args.save_preview:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    psnr_avg_list = []
    ssim_avg_list = []
    psnr_sharp_list = []
    ssim_sharp_list = []

    for center_path in centers:
        rel = center_path.relative_to(input_root)
        seq_input_dir = center_path.parent
        seq_target_dir = target_root / rel.parent

        frames = []
        for i in range(1, 8):
            p = seq_input_dir / f"im{i}.png"
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read {p}")
            frames.append(bgr_to_y_float01(img))

        gt_path = seq_target_dir / "im4.png"
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if gt_img is None:
            raise FileNotFoundError(f"Failed to read {gt_path}")
        gt_y = bgr_to_y_float01(gt_img)

        avg = temporal_weighted_average(frames, center=3, weights=weights)
        sharp = unsharp_mask(avg, sigma=args.sigma, amount=args.amount)

        p_avg = psnr_y(avg, gt_y)
        s_avg = ssim_y(avg, gt_y)
        p_sh = psnr_y(sharp, gt_y)
        s_sh = ssim_y(sharp, gt_y)

        psnr_avg_list.append(p_avg)
        ssim_avg_list.append(s_avg)
        psnr_sharp_list.append(p_sh)
        ssim_sharp_list.append(s_sh)

        rows.append([str(rel.parent), p_avg, s_avg, p_sh, s_sh])

        if args.save_preview:
            key = rel.parent.as_posix().replace("/", "_")
            cv2.imwrite(str(args.preview_dir / f"{key}_avg.png"), (avg * 255.0).astype(np.uint8))
            cv2.imwrite(str(args.preview_dir / f"{key}_sharp.png"), (sharp * 255.0).astype(np.uint8))

    mean_avg_psnr = float(np.mean(psnr_avg_list))
    mean_avg_ssim = float(np.mean(ssim_avg_list))
    mean_sh_psnr = float(np.mean(psnr_sharp_list))
    mean_sh_ssim = float(np.mean(ssim_sharp_list))

    print(f"[temporal_avg] PSNR={mean_avg_psnr:.4f} SSIM={mean_avg_ssim:.4f}")
    print(f"[temporal_avg_unsharp] PSNR={mean_sh_psnr:.4f} SSIM={mean_sh_ssim:.4f}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence",
            "psnr_temporal_avg",
            "ssim_temporal_avg",
            "psnr_temporal_unsharp",
            "ssim_temporal_unsharp",
        ])
        writer.writerows(rows)
        writer.writerow(["MEAN", mean_avg_psnr, mean_avg_ssim, mean_sh_psnr, mean_sh_ssim])

    print(f"Saved temporal metrics to: {args.csv_out}")


if __name__ == "__main__":
    main()
