#!/usr/bin/env python3
"""
Quick FUGR parameter sweep.

This script evaluates many FUGR-VSR parameter combinations without saving
100 output frames for every combination. It is much faster and avoids wasting disk I/O.

Run from:
  /home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part3
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def gray(img):
    return cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def norm01(x, eps=1e-8):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


def psnr(x, y):
    if x.shape[:2] != y.shape[:2]:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def ssim_rgb(x, y):
    if x.shape[:2] != y.shape[:2]:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        a = x[..., ch].astype(np.float32)
        b = y[..., ch].astype(np.float32)
        ma = cv2.GaussianBlur(a, (11, 11), 1.5)
        mb = cv2.GaussianBlur(b, (11, 11), 1.5)
        va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - ma * ma
        vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mb * mb
        vab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - ma * mb
        vals.append(float(np.mean(((2 * ma * mb + c1) * (2 * vab + c2)) /
                                  ((ma * ma + mb * mb + c1) * (va + vb + c2) + 1e-12))))
    return float(np.mean(vals))


def sharpness(img):
    return float(np.var(cv2.Laplacian(gray(img), cv2.CV_32F)))


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def texture(basic):
    g = gray(basic)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return norm01(cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2))


def disagreement(basic, gan):
    return cv2.GaussianBlur(np.mean(np.abs(basic - gan), axis=2), (0, 0), 1.5)


def temp_risk(res_prev, res_cur, res_next):
    if res_prev is None or res_next is None:
        return np.zeros(res_cur.shape[:2], np.float32)
    pred = 0.5 * (res_prev + res_next)
    return cv2.GaussianBlur(np.mean(np.abs(res_cur - pred), axis=2), (0, 0), 1.2)


def masks(basic, gan, res_prev, res_cur, res_next, max_alpha, tau_dis, tau_temp, temporal):
    tex = texture(basic)
    dis = disagreement(basic, gan)
    rel_dis = np.exp(-dis / tau_dis)
    if temporal:
        tr = temp_risk(res_prev, res_cur, res_next)
        rel_temp = np.exp(-tr / tau_temp)
    else:
        tr = np.zeros_like(dis)
        rel_temp = np.ones_like(dis)
    alpha = max_alpha * tex * rel_dis * rel_temp
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
    return np.clip(alpha, 0, max_alpha)


def rgb_blend(basic, gan, alpha):
    return np.clip((1 - alpha[..., None]) * basic + alpha[..., None] * gan, 0, 1)


def fugr(basic, gan, alpha, sigma, detail_strength):
    detail = highpass(gan, sigma) - highpass(basic, sigma)
    return np.clip(basic + detail_strength * alpha[..., None] * detail, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic_dir", default="basicvsr_result/000")
    ap.add_argument("--gan_dir", default="vsrgan_result/000")
    ap.add_argument("--gt_dir", default="/home/schung760/shared_data/project1/val/val_sharp/000")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.6])
    ap.add_argument("--tau_dis", type=float, default=0.08)
    ap.add_argument("--tau_temp", type=float, default=0.04)
    args = ap.parse_args()

    basic_dir = Path(args.basic_dir)
    gan_dir = Path(args.gan_dir)
    gt_dir = Path(args.gt_dir)
    names = sorted([p.name for p in gt_dir.glob("*.png") if (basic_dir / p.name).exists() and (gan_dir / p.name).exists()])
    if not names:
        raise RuntimeError("No matched frames found.")

    print(f"Loading {len(names)} frames...")
    basics = {n: read_rgb(basic_dir / n) for n in names}
    gans = {n: read_rgb(gan_dir / n) for n in names}
    gts = {n: read_rgb(gt_dir / n) for n in names}

    rows = []
    total = len(args.alphas) * len(args.strengths) * len(args.sigmas)
    k = 0

    for sigma in args.sigmas:
        residuals = {n: highpass(gans[n], sigma) - highpass(basics[n], sigma) for n in names}

        for alpha_max in args.alphas:
            for strength in args.strengths:
                k += 1
                print(f"[{k}/{total}] alpha={alpha_max}, strength={strength}, sigma={sigma}", flush=True)

                vals = {
                    "RGB-Hybrid": [],
                    "FUGR-no-temporal": [],
                    "FUGR-temporal": [],
                }

                for i, n in enumerate(names):
                    b, g, gt = basics[n], gans[n], gts[n]
                    rp = residuals[names[i - 1]] if i > 0 else None
                    rc = residuals[n]
                    rn = residuals[names[i + 1]] if i + 1 < len(names) else None

                    a_no = masks(b, g, rp, rc, rn, alpha_max, args.tau_dis, args.tau_temp, False)
                    a_t = masks(b, g, rp, rc, rn, alpha_max, args.tau_dis, args.tau_temp, True)

                    outputs = {
                        "RGB-Hybrid": rgb_blend(b, g, a_t),
                        "FUGR-no-temporal": fugr(b, g, a_no, sigma, strength),
                        "FUGR-temporal": fugr(b, g, a_t, sigma, strength),
                    }

                    for method, img in outputs.items():
                        vals[method].append((psnr(img, gt), ssim_rgb(img, gt), sharpness(img)))

                for method, metrics in vals.items():
                    arr = np.asarray(metrics, dtype=np.float64)
                    rows.append({
                        "max_alpha": alpha_max,
                        "detail_strength": strength,
                        "hp_sigma": sigma,
                        "method": method,
                        "psnr": arr[:, 0].mean(),
                        "ssim": arr[:, 1].mean(),
                        "laplacian_sharpness": arr[:, 2].mean(),
                    })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["psnr"], r["ssim"]), reverse=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["max_alpha", "detail_strength", "hp_sigma", "method", "psnr", "ssim", "laplacian_sharpness"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "max_alpha": r["max_alpha"],
                "detail_strength": r["detail_strength"],
                "hp_sigma": r["hp_sigma"],
                "method": r["method"],
                "psnr": f"{r['psnr']:.6f}",
                "ssim": f"{r['ssim']:.6f}",
                "laplacian_sharpness": f"{r['laplacian_sharpness']:.8f}",
            })

    print("\nSaved:", out_csv)
    print("Top 15:")
    for r in rows[:15]:
        print(f"{r['method']}: alpha={r['max_alpha']}, strength={r['detail_strength']}, sigma={r['hp_sigma']}, "
              f"PSNR={r['psnr']:.4f}, SSIM={r['ssim']:.4f}, sharp={r['laplacian_sharpness']:.8f}")


if __name__ == "__main__":
    main()
