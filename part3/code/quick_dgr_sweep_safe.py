import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


INPUT_DIR = Path("/home/schung760/my_storage2_1T/AIAA3201_Part3_outputs/DirectionB/expanded_selected_conservative")
OUT_DIR = Path("/home/schung760/my_storage2_1T/AIAA3201_Part3_outputs/DirectionB/dgr_from_expanded_conservative_safe")

ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
SIGMAS = [0.8, 1.0, 1.6, 2.4]
TAUS = [0.05, 0.08, 0.12]


def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_rgb(p, img):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
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


def dgr_fugr(fugr, cn, alpha, sigma, tau):
    dis = cv2.GaussianBlur(np.mean(np.abs(fugr - cn), axis=2), (0, 0), 1.5)
    mask = texture_map(fugr) * np.exp(-dis / tau)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 1.0), 0, 1)

    residual = highpass(cn, sigma) - highpass(fugr, sigma)
    out = np.clip(fugr + alpha * mask[..., None] * residual, 0, 1)
    return out, mask


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


def collect_samples():
    frames = INPUT_DIR / "frames"
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
                "basic_path": basic_path if basic_path.exists() else fugr_path,
                "fugr_path": fugr_path,
                "cn_path": cn_path,
                "gt_path": gt_path,
            })

    return samples


def summarize_existing(samples, method):
    psnrs, ssims, sharps = [], [], []
    by_seq = defaultdict(list)
    gt_by_seq = defaultdict(list)

    for s in samples:
        if method == "BasicVSR":
            img = read_rgb(s["basic_path"])
        elif method == "FUGR-C":
            img = read_rgb(s["fugr_path"])
        elif method == "ControlNet-FUGR":
            img = read_rgb(s["cn_path"])
        else:
            raise ValueError(method)

        gt = read_rgb(s["gt_path"])

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


def summarize_dgr(samples, alpha, sigma, tau):
    psnrs, ssims, sharps, masks = [], [], [], []
    by_seq = defaultdict(list)
    gt_by_seq = defaultdict(list)

    for s in samples:
        fugr = read_rgb(s["fugr_path"])
        cn = read_rgb(s["cn_path"])
        gt = read_rgb(s["gt_path"])

        img, mask = dgr_fugr(fugr, cn, alpha, sigma, tau)

        psnrs.append(psnr(img, gt))
        ssims.append(ssim_rgb(img, gt))
        sharps.append(sharpness(img))
        masks.append(float(mask.mean()))

        by_seq[s["seq"]].append((s["frame"], img))
        gt_by_seq[s["seq"]].append((s["frame"], gt))

    tdes = []
    for seq in by_seq:
        imgs = [x[1] for x in sorted(by_seq[seq], key=lambda z: z[0])]
        gts = [x[1] for x in sorted(gt_by_seq[seq], key=lambda z: z[0])]
        tdes.append(tde(imgs, gts))

    return {
        "method": f"DGR-a{alpha:.2f}-s{sigma:.1f}-t{tau:.2f}",
        "num_frames": len(samples),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "laplacian_sharpness": float(np.mean(sharps)),
        "tde": float(np.mean(tdes)),
        "alpha": alpha,
        "sigma": sigma,
        "tau": tau,
        "mask_mean": float(np.mean(masks)),
    }


def save_best_outputs(samples, best):
    panels_dir = OUT_DIR / "panels"
    frames_dir = OUT_DIR / "best_frames"

    count = 0
    for s in samples:
        basic = read_rgb(s["basic_path"])
        fugr = read_rgb(s["fugr_path"])
        cn = read_rgb(s["cn_path"])
        gt = read_rgb(s["gt_path"])

        img, mask = dgr_fugr(fugr, cn, best["alpha"], best["sigma"], best["tau"])

        save_rgb(frames_dir / s["seq"] / f"{s['frame']}_dgr.png", img)
        save_rgb(frames_dir / s["seq"] / f"{s['frame']}_mask.png", colorize(mask))

        if count < 16:
            make_panel(
                panels_dir / f"panel_{s['seq']}_{s['frame']}.png",
                [fugr, cn, img, gt, err_map(fugr, gt), err_map(cn, gt), err_map(img, gt), colorize(mask)],
                ["FUGR-C", "CN-FUGR", "DGR", "GT", "FUGR Err", "CN Err", "DGR Err", "DGR Mask"]
            )
            count += 1


def main():
    metrics_dir = OUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples()
    print("Found samples:", len(samples), flush=True)

    rows = []
    for method in ["BasicVSR", "FUGR-C", "ControlNet-FUGR"]:
        print("Summarizing", method, flush=True)
        rows.append(summarize_existing(samples, method))

    total = len(ALPHAS) * len(SIGMAS) * len(TAUS)
    k = 0

    for alpha in ALPHAS:
        for sigma in SIGMAS:
            for tau in TAUS:
                k += 1
                print(f"[{k}/{total}] alpha={alpha}, sigma={sigma}, tau={tau}", flush=True)
                rows.append(summarize_dgr(samples, alpha, sigma, tau))

    rows.sort(key=lambda r: (r["psnr"], r["ssim"]), reverse=True)

    csv_path = metrics_dir / "dgr_config_summary.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["method", "num_frames", "psnr", "ssim", "laplacian_sharpness", "tde", "alpha", "sigma", "tau", "mask_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_dgr = next(r for r in rows if r["method"].startswith("DGR"))

    txt_path = metrics_dir / "dgr_best_summary.txt"
    with txt_path.open("w") as f:
        f.write("Direction B10: Memory-safe Diffusion-Guided Residual Hybrid\n")
        f.write(f"input_dir: {INPUT_DIR}\n")
        f.write(f"num_samples: {len(samples)}\n\n")
        f.write("Top 20 methods/configs by PSNR:\n")
        for r in rows[:20]:
            f.write(
                f"{r['method']},{r['num_frames']},{r['psnr']:.6f},"
                f"{r['ssim']:.6f},{r['laplacian_sharpness']:.8f},{r['tde']:.8f},"
                f"{r.get('alpha','')},{r.get('sigma','')},{r.get('tau','')},{r.get('mask_mean','')}\n"
            )
        f.write("\nBest DGR:\n")
        f.write(str(best_dgr) + "\n")

    print(txt_path.read_text(), flush=True)

    save_best_outputs(samples, best_dgr)
    print("Saved:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
