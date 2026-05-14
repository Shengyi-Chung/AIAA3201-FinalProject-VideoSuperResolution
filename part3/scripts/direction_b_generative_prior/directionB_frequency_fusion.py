import argparse
import csv
from pathlib import Path
from collections import defaultdict

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


def gray(img):
    return cv2.cvtColor(
        np.clip(img * 255, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0


def norm01(x, eps=1e-8):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


def colorize(x):
    u8 = np.clip(norm01(x) * 255, 0, 255).astype(np.uint8)
    c = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def texture_map(img):
    g = gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2)
    return norm01(grad)


def disagreement_map(a, b):
    return cv2.GaussianBlur(np.mean(np.abs(a - b), axis=2), (0, 0), 1.5)


def rgb_hf_fusion(fugr, cn, beta, sigma):
    residual = highpass(cn, sigma) - highpass(fugr, sigma)
    return np.clip(fugr + beta * residual, 0, 1)


def rgb_masked_hf_fusion(fugr, cn, beta, sigma, tau):
    dis = disagreement_map(fugr, cn)
    mask = texture_map(fugr) * np.exp(-dis / tau)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 1.0), 0, 1)
    residual = highpass(cn, sigma) - highpass(fugr, sigma)
    out = np.clip(fugr + beta * mask[..., None] * residual, 0, 1)
    return out, mask


def to_ycrcb(img):
    u8 = np.clip(img * 255, 0, 255).round().astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    return ycc


def from_ycrcb(ycc):
    u8 = np.clip(ycc * 255, 0, 255).round().astype(np.uint8)
    rgb = cv2.cvtColor(u8, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return rgb


def y_hf_fusion(fugr, cn, beta, sigma):
    fy = to_ycrcb(fugr)
    cy = to_ycrcb(cn)

    y_f = fy[..., 0]
    y_c = cy[..., 0]

    residual = highpass(y_c, sigma) - highpass(y_f, sigma)
    y_out = np.clip(y_f + beta * residual, 0, 1)

    out_ycc = fy.copy()
    out_ycc[..., 0] = y_out
    return from_ycrcb(out_ycc)


def y_masked_hf_fusion(fugr, cn, beta, sigma, tau):
    fy = to_ycrcb(fugr)
    cy = to_ycrcb(cn)

    y_f = fy[..., 0]
    y_c = cy[..., 0]

    dis = cv2.GaussianBlur(np.abs(y_f - y_c), (0, 0), 1.5)
    mask = texture_map(fugr) * np.exp(-dis / tau)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 1.0), 0, 1)

    residual = highpass(y_c, sigma) - highpass(y_f, sigma)
    y_out = np.clip(y_f + beta * mask * residual, 0, 1)

    out_ycc = fy.copy()
    out_ycc[..., 0] = y_out
    return from_ycrcb(out_ycc), mask


def psnr(x, y):
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def ssim_rgb(x, y):
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

        ssim = ((2 * ma * mb + c1) * (2 * vab + c2)) / (
            (ma * ma + mb * mb + c1) * (va + vb + c2) + 1e-12
        )
        vals.append(float(np.mean(ssim)))
    return float(np.mean(vals))


def sharpness(img):
    return float(np.var(cv2.Laplacian(gray(img), cv2.CV_32F)))


def tde(imgs, gts):
    if len(imgs) < 2:
        return 0.0
    vals = []
    for i in range(1, len(imgs)):
        vals.append(float(np.mean(np.abs((imgs[i] - imgs[i - 1]) - (gts[i] - gts[i - 1])))))
    return float(np.mean(vals))


def resize_h(img, target_h=220):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)


def err_map(img, gt):
    return colorize(np.mean(np.abs(img - gt), axis=2))


def make_panel(path, imgs, titles):
    imgs = [resize_h(x, 220) for x in imgs]
    title_h = 34
    h = imgs[0].shape[0]
    w = sum(x.shape[1] for x in imgs)
    canvas = np.ones((h + title_h, w, 3), dtype=np.uint8) * 255

    x0 = 0
    for img, title in zip(imgs, titles):
        ww = img.shape[1]
        canvas[title_h:, x0:x0 + ww] = np.clip(img * 255, 0, 255).round().astype(np.uint8)
        cv2.putText(canvas, title, (x0 + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        x0 += ww

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def collect_samples(input_dir):
    frames = Path(input_dir) / "frames"
    samples = []
    for fugr_path in sorted(frames.glob("*/*_fugr.png")):
        seq = fugr_path.parent.name
        stem = fugr_path.name.replace("_fugr.png", "")

        cn_path = fugr_path.parent / f"{stem}_controlnet_fugr.png"
        gt_path = fugr_path.parent / f"{stem}_gt.png"
        basic_path = fugr_path.parent / f"{stem}_basic.png"

        if cn_path.exists() and gt_path.exists():
            samples.append({
                "seq": seq,
                "frame": stem,
                "basic": read_rgb(basic_path) if basic_path.exists() else read_rgb(fugr_path),
                "fugr": read_rgb(fugr_path),
                "cn": read_rgb(cn_path),
                "gt": read_rgb(gt_path),
            })
    return samples


def summarize(samples, method, get_img):
    psnrs, ssims, sharps = [], [], []
    by_seq = defaultdict(list)
    gt_by_seq = defaultdict(list)

    for s in samples:
        img = get_img(s)
        gt = s["gt"]

        psnrs.append(psnr(img, gt))
        ssims.append(ssim_rgb(img, gt))
        sharps.append(sharpness(img))

        by_seq[s["seq"]].append((s["frame"], img))
        gt_by_seq[s["seq"]].append((s["frame"], gt))

    tdes = []
    for seq in by_seq:
        imgs = [x[1] for x in sorted(by_seq[seq], key=lambda z: z[0])]
        gts = [x[1] for x in sorted(gt_by_seq[seq], key=lambda z: z[0])]
        tdes.append(tde(imgs, gts))

    return {
        "method": method,
        "num_frames": len(samples),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "laplacian_sharpness": float(np.mean(sharps)),
        "tde": float(np.mean(tdes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.4, 0.8, 1.0, 1.6, 2.4])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.03, 0.05, 0.08, 0.12])
    parser.add_argument("--panel_limit", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    panels_dir = out_dir / "panels"
    frames_dir = out_dir / "best_frames"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.input_dir)
    print("Found samples:", len(samples), flush=True)

    rows = []
    rows.append(summarize(samples, "BasicVSR", lambda s: s["basic"]))
    rows.append(summarize(samples, "FUGR-C", lambda s: s["fugr"]))
    rows.append(summarize(samples, "ControlNet-FUGR", lambda s: s["cn"]))

    cache_best = {}

    total = len(args.betas) * len(args.sigmas) * (2 + 2 * len(args.taus))
    k = 0

    for beta in args.betas:
        for sigma in args.sigmas:
            k += 1
            method = f"RGB-HF-b{beta:.2f}-s{sigma:.1f}"
            print(f"[{k}/{total}] {method}", flush=True)
            rows.append(summarize(samples, method, lambda s, b=beta, sg=sigma: rgb_hf_fusion(s["fugr"], s["cn"], b, sg)))

            k += 1
            method = f"Y-HF-b{beta:.2f}-s{sigma:.1f}"
            print(f"[{k}/{total}] {method}", flush=True)
            rows.append(summarize(samples, method, lambda s, b=beta, sg=sigma: y_hf_fusion(s["fugr"], s["cn"], b, sg)))

            for tau in args.taus:
                k += 1
                method = f"RGB-MaskHF-b{beta:.2f}-s{sigma:.1f}-t{tau:.2f}"
                print(f"[{k}/{total}] {method}", flush=True)
                rows.append(summarize(samples, method, lambda s, b=beta, sg=sigma, t=tau: rgb_masked_hf_fusion(s["fugr"], s["cn"], b, sg, t)[0]))

                k += 1
                method = f"Y-MaskHF-b{beta:.2f}-s{sigma:.1f}-t{tau:.2f}"
                print(f"[{k}/{total}] {method}", flush=True)
                rows.append(summarize(samples, method, lambda s, b=beta, sg=sigma, t=tau: y_masked_hf_fusion(s["fugr"], s["cn"], b, sg, t)[0]))

    rows.sort(key=lambda r: (r["psnr"], r["ssim"]), reverse=True)

    csv_path = metrics_dir / "frequency_fusion_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "num_frames", "psnr", "ssim", "laplacian_sharpness", "tde"])
        writer.writeheader()
        writer.writerows(rows)

    best_nonbaseline = next(r for r in rows if r["method"] not in ["BasicVSR", "FUGR-C", "ControlNet-FUGR"])

    txt_path = metrics_dir / "frequency_fusion_best_summary.txt"
    with txt_path.open("w") as f:
        f.write("Direction B13: Frequency / Luminance-Constrained Fusion\n")
        f.write(f"input_dir: {args.input_dir}\n")
        f.write(f"num_samples: {len(samples)}\n\n")
        f.write("Top 25 methods/configs by PSNR:\n")
        for r in rows[:25]:
            f.write(f"{r['method']},{r['num_frames']},{r['psnr']:.6f},{r['ssim']:.6f},{r['laplacian_sharpness']:.8f},{r['tde']:.8f}\n")
        f.write("\nBest non-baseline:\n")
        f.write(str(best_nonbaseline) + "\n")

    print(txt_path.read_text(), flush=True)

    best_method = best_nonbaseline["method"]

    def apply_best(s):
        parts = best_method.split("-")
        if best_method.startswith("RGB-HF"):
            beta = float(parts[2][1:])
            sigma = float(parts[3][1:])
            return rgb_hf_fusion(s["fugr"], s["cn"], beta, sigma), None
        if best_method.startswith("Y-HF"):
            beta = float(parts[2][1:])
            sigma = float(parts[3][1:])
            return y_hf_fusion(s["fugr"], s["cn"], beta, sigma), None
        if best_method.startswith("RGB-MaskHF"):
            beta = float(parts[2][1:])
            sigma = float(parts[3][1:])
            tau = float(parts[4][1:])
            return rgb_masked_hf_fusion(s["fugr"], s["cn"], beta, sigma, tau)
        if best_method.startswith("Y-MaskHF"):
            beta = float(parts[2][1:])
            sigma = float(parts[3][1:])
            tau = float(parts[4][1:])
            return y_masked_hf_fusion(s["fugr"], s["cn"], beta, sigma, tau)
        raise ValueError(best_method)

    count = 0
    for s in samples:
        best_img, mask = apply_best(s)
        save_rgb(frames_dir / s["seq"] / f"{s['frame']}_bestfreq.png", best_img)

        if count < args.panel_limit:
            if mask is None:
                mask_img = np.zeros_like(gray(s["fugr"]))
            else:
                mask_img = mask

            make_panel(
                panels_dir / f"panel_{s['seq']}_{s['frame']}.png",
                [
                    s["fugr"],
                    s["cn"],
                    best_img,
                    s["gt"],
                    err_map(s["fugr"], s["gt"]),
                    err_map(s["cn"], s["gt"]),
                    err_map(best_img, s["gt"]),
                    colorize(mask_img),
                ],
                ["FUGR-C", "CN-FUGR", "FreqFusion", "GT", "FUGR Err", "CN Err", "Fusion Err", "Mask"]
            )
            count += 1

    print("Saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
