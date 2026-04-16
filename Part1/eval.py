from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity

from dataset import build_frame_pairs, default_split_paths, load_bgr, load_y_channel
from train_srcnn import SRCNN


def psnr_y(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0.0:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_y(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(structural_similarity(pred, gt, data_range=1.0))


def load_srcnn(ckpt: Path, device: torch.device) -> nn.Module:
    model = SRCNN().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def degrade_then_upsample(hr_bgr: np.ndarray, method: str, scale: int = 4) -> np.ndarray:
    h, w = hr_bgr.shape[:2]
    lr = cv2.resize(hr_bgr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

    if method == "bicubic":
        interp = cv2.INTER_CUBIC
    elif method == "lanczos":
        interp = cv2.INTER_LANCZOS4
    else:
        raise ValueError(f"Unsupported method: {method}")

    return cv2.resize(lr, (w, h), interpolation=interp)


def eval_interp_from_hr(pairs: Iterable[tuple[Path, Path]], method: str) -> tuple[float, float, float]:
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    t0 = perf_counter()
    count = 0

    for _, hr in pairs:
        hr_bgr = load_bgr(hr)
        pred_bgr = degrade_then_upsample(hr_bgr, method=method, scale=4)

        pred_y = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
        gt_y = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0

        psnr_list.append(psnr_y(pred_y, gt_y))
        ssim_list.append(ssim_y(pred_y, gt_y))
        count += 1

    elapsed = perf_counter() - t0
    fps = count / elapsed if elapsed > 0 else 0.0
    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), fps


def eval_input_bicubic(pairs: Iterable[tuple[Path, Path]]) -> tuple[float, float, float]:
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    t0 = perf_counter()
    count = 0

    for lr_up, hr in pairs:
        pred = load_y_channel(lr_up)
        gt = load_y_channel(hr)
        psnr_list.append(psnr_y(pred, gt))
        ssim_list.append(ssim_y(pred, gt))
        count += 1

    elapsed = perf_counter() - t0
    fps = count / elapsed if elapsed > 0 else 0.0
    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), fps


def eval_srcnn_pairs(
    pairs: Iterable[tuple[Path, Path]],
    model: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    t0 = perf_counter()
    count = 0

    with torch.no_grad():
        for lr_up, hr in pairs:
            inp = load_y_channel(lr_up)
            gt = load_y_channel(hr)

            x = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)

            psnr_list.append(psnr_y(pred, gt))
            ssim_list.append(ssim_y(pred, gt))
            count += 1

    elapsed = perf_counter() - t0
    fps = count / elapsed if elapsed > 0 else 0.0
    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Part1 baselines")
    parser.add_argument("--project1-root", type=Path, default=Path("/home/schung760/shared_data/project1"))
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument(
        "--srcnn-ckpt",
        type=Path,
        default=Path("/home/schung760/AIAA3201-FinalProject-VideoSuperResolution/Part1/checkpoints/srcnn_best.pt"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("/home/schung760/AIAA3201-FinalProject-VideoSuperResolution/Part1/results_part1.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_paths = default_split_paths(args.project1_root)[args.split]
    pairs = build_frame_pairs(split_paths.hr_root, split_paths.lr_bicubic_root)

    rows: list[tuple[str, float, float, float]] = []

    bic_psnr, bic_ssim, bic_fps = eval_interp_from_hr(pairs, method="bicubic")
    rows.append(("bicubic", bic_psnr, bic_ssim, bic_fps))
    print(f"[bicubic] PSNR={bic_psnr:.4f} SSIM={bic_ssim:.4f} FPS={bic_fps:.2f}")

    lan_psnr, lan_ssim, lan_fps = eval_interp_from_hr(pairs, method="lanczos")
    rows.append(("lanczos", lan_psnr, lan_ssim, lan_fps))
    print(f"[lanczos] PSNR={lan_psnr:.4f} SSIM={lan_ssim:.4f} FPS={lan_fps:.2f}")

    inp_psnr, inp_ssim, inp_fps = eval_input_bicubic(pairs)
    rows.append(("input_bicubic", inp_psnr, inp_ssim, inp_fps))
    print(f"[input_bicubic] PSNR={inp_psnr:.4f} SSIM={inp_ssim:.4f} FPS={inp_fps:.2f}")

    if args.srcnn_ckpt.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_srcnn(args.srcnn_ckpt, device)
        psnr, ssim, fps = eval_srcnn_pairs(pairs, model, device)
        rows.append(("srcnn", psnr, ssim, fps))
        print(f"[srcnn] PSNR={psnr:.4f} SSIM={ssim:.4f} FPS={fps:.2f}")
    else:
        print(f"Skip SRCNN: checkpoint not found at {args.srcnn_ckpt}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "psnr_y", "ssim_y", "fps"])
        writer.writerows(rows)

    print(f"Saved metrics to: {args.csv_out}")


if __name__ == "__main__":
    main()
