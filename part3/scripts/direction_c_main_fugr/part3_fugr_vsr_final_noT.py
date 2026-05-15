#!/usr/bin/env python3
"""
Part3: Frequency-aware Uncertainty-Guided Residual Refinement (FUGR-VSR)

This script fuses existing BasicVSR and VSRGAN output PNG frames.
It saves FUGR-no-temporal frames, per-frame metrics, summary text, and qualitative panels.
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


def save_rgb(p, img):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(img * 255, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(p), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def norm01(x, eps=1e-8):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


def gray(img):
    return cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


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
    return np.clip(alpha, 0, max_alpha), tex, dis, tr


def rgb_blend(basic, gan, alpha):
    return np.clip((1 - alpha[..., None]) * basic + alpha[..., None] * gan, 0, 1)


def fugr(basic, gan, alpha, sigma, detail_strength):
    detail = highpass(gan, sigma) - highpass(basic, sigma)
    return np.clip(basic + detail_strength * alpha[..., None] * detail, 0, 1)


def colorize(x):
    c = cv2.applyColorMap((norm01(x) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def panel(path, imgs, titles):
    target_h = 300
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        resized.append(cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA))
    title_h = 36
    total_w = sum(im.shape[1] for im in resized)
    canvas = np.ones((target_h + title_h, total_w, 3), np.uint8) * 255
    x = 0
    for im, title in zip(resized, titles):
        w = im.shape[1]
        canvas[title_h:, x:x+w] = np.clip(im * 255, 0, 255).round().astype(np.uint8)
        cv2.putText(canvas, title, (x + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        x += w
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def temporal_difference_error(frames, gt_frames):
    if len(frames) < 2:
        return float("nan")
    vals = []
    for i in range(1, len(frames)):
        vals.append(float(np.mean(np.abs((frames[i] - frames[i-1]) - (gt_frames[i] - gt_frames[i-1])))))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic_dir", required=True)
    ap.add_argument("--gan_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--summary_path", required=True)
    ap.add_argument("--max_alpha", type=float, default=0.25)
    ap.add_argument("--tau_dis", type=float, default=0.08)
    ap.add_argument("--tau_temp", type=float, default=0.04)
    ap.add_argument("--hp_sigma", type=float, default=1.6)
    ap.add_argument("--detail_strength", type=float, default=1.2)
    ap.add_argument("--panel_every", type=int, default=25)
    args = ap.parse_args()

    basic_dir, gan_dir, gt_dir = Path(args.basic_dir), Path(args.gan_dir), Path(args.gt_dir)
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    csv_path, summary_path = Path(args.csv_path), Path(args.summary_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    names = sorted([p.name for p in gt_dir.glob("*.png") if (basic_dir / p.name).exists() and (gan_dir / p.name).exists()])
    if not names:
        raise RuntimeError("No matched frames found. Check basic_dir, gan_dir, and gt_dir.")
    print("Matched frames:", len(names))

    basics, gans, residuals = {}, {}, {}
    for n in names:
        b, g = read_rgb(basic_dir / n), read_rgb(gan_dir / n)
        basics[n], gans[n] = b, g
        residuals[n] = highpass(g, args.hp_sigma) - highpass(b, args.hp_sigma)

    methods = ["BasicVSR", "VSRGAN", "RGB-Hybrid", "FUGR-no-temporal", "FUGR-temporal"]
    vals = {m: [] for m in methods}
    outputs_by_method = {m: [] for m in methods}
    gt_list = []
    rows = []

    for i, n in enumerate(names):
        b, g, gt = basics[n], gans[n], read_rgb(gt_dir / n)
        rp = residuals[names[i - 1]] if i > 0 else None
        rc = residuals[n]
        rn = residuals[names[i + 1]] if i + 1 < len(names) else None

        a0, tex, dis, tr0 = masks(b, g, rp, rc, rn, args.max_alpha, args.tau_dis, args.tau_temp, False)
        at, tex, dis, tr = masks(b, g, rp, rc, rn, args.max_alpha, args.tau_dis, args.tau_temp, True)

        outs = {
            "BasicVSR": b,
            "VSRGAN": g,
            "RGB-Hybrid": rgb_blend(b, g, at),
            "FUGR-no-temporal": fugr(b, g, a0, args.hp_sigma, args.detail_strength),
            "FUGR-temporal": fugr(b, g, at, args.hp_sigma, args.detail_strength),
        }

        save_rgb(out_dir / n, outs["FUGR-no-temporal"])
        gt_list.append(gt)

        row = {
            "frame": n,
            "alpha_noT_mean": float(a0.mean()),
            "alpha_T_mean": float(at.mean()),
            "alpha_T_max": float(at.max()),
            "disagreement_mean": float(dis.mean()),
            "temporal_risk_mean": float(tr.mean()),
        }
        for m, im in outs.items():
            p, s, sh = psnr(im, gt), ssim_rgb(im, gt), sharpness(im)
            vals[m].append((p, s, sh))
            outputs_by_method[m].append(im)
            row[f"{m}_psnr"], row[f"{m}_ssim"], row[f"{m}_sharp"] = p, s, sh
        rows.append(row)

        if args.panel_every > 0 and i % args.panel_every == 0:
            panel(
                fig_dir / f"fugr_panel_{Path(n).stem}.png",
                [
                    b,
                    g,
                    outs["RGB-Hybrid"],
                    outs["FUGR-no-temporal"],
                    outs["FUGR-temporal"],
                    colorize(at),
                    colorize(dis),
                    colorize(tr),
                    gt,
                ],
                ["BasicVSR", "VSRGAN", "RGB", "FUGR-noT", "FUGR-T", "Alpha", "Disagree", "TempRisk", "GT"],
            )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Part3: Frequency-aware Uncertainty-Guided Residual Refinement (FUGR-VSR)",
        f"Matched frames: {len(names)}",
        f"max_alpha: {args.max_alpha}",
        f"tau_dis: {args.tau_dis}",
        f"tau_temp: {args.tau_temp}",
        f"hp_sigma: {args.hp_sigma}",
        f"detail_strength: {args.detail_strength}",
        "",
        "method,psnr,ssim,laplacian_sharpness,tde",
    ]
    for m in methods:
        arr = np.array(vals[m], dtype=np.float64)
        tde = temporal_difference_error(outputs_by_method[m], gt_list)
        lines.append(f"{m},{arr[:,0].mean():.4f},{arr[:,1].mean():.4f},{arr[:,2].mean():.8f},{tde:.8f}")

    summary_path.write_text("\n".join(lines) + "\n")
    print(summary_path.read_text())
    print("Saved output:", out_dir)
    print("Saved figures:", fig_dir)
    print("Saved csv:", csv_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
