#!/usr/bin/env python3
import argparse
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

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma)

def rgb_hf(fugr, cn, beta, sigma):
    return np.clip(fugr + beta * (highpass(cn, sigma) - highpass(fugr, sigma)), 0, 1)

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))

def colorize(x):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    y = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    cm = cv2.applyColorMap((y * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def err_map(a, b):
    return colorize(np.mean(np.abs(a - b), axis=2))

def crop_around_edge(gt, crop_size):
    mag = cv2.GaussianBlur(
        np.sqrt(cv2.Sobel(gray(gt), cv2.CV_32F, 1, 0, ksize=3) ** 2 +
                cv2.Sobel(gray(gt), cv2.CV_32F, 0, 1, ksize=3) ** 2),
        (0, 0), 4.0
    )
    h, w = mag.shape
    y, x = np.unravel_index(np.argmax(mag), mag.shape)
    c = min(crop_size, h, w)
    x0 = max(0, min(w - c, x - c // 2))
    y0 = max(0, min(h - c, y - c // 2))
    return int(x0), int(y0), int(c), int(c)

def crop(img, box):
    x, y, w, h = box
    return img[y:y+h, x:x+w]

def resize_h(img, h=220):
    H, W = img.shape[:2]
    return cv2.resize(img, (int(W * h / H), h), interpolation=cv2.INTER_AREA)

def panel(path, imgs, titles):
    imgs = [resize_h(x) for x in imgs]
    title_h = 34
    H = imgs[0].shape[0]
    W = sum(x.shape[1] for x in imgs)
    canvas = np.ones((H + title_h, W, 3), dtype=np.uint8) * 255
    x0 = 0
    for im, title in zip(imgs, titles):
        w = im.shape[1]
        canvas[title_h:, x0:x0+w] = np.clip(im * 255, 0, 255).round().astype(np.uint8)
        cv2.putText(canvas, title, (x0 + 5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
        x0 += w
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def collect(input_dir):
    root = Path(input_dir) / "frames"
    samples = []
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        for fp in sorted(sd.glob("*_fugr.png")):
            frame = fp.name.replace("_fugr.png", "")
            cn = sd / f"{frame}_controlnet_fugr.png"
            gt = sd / f"{frame}_gt.png"
            basic = sd / f"{frame}_basic.png"
            if cn.exists() and gt.exists():
                samples.append({
                    "seq": sd.name,
                    "frame": frame,
                    "basic": read_rgb(basic) if basic.exists() else read_rgb(fp),
                    "fugr": read_rgb(fp),
                    "cn": read_rgb(cn),
                    "gt": read_rgb(gt),
                })
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.60)
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--panel_count", type=int, default=12)
    ap.add_argument("--crop_size", type=int, default=160)
    args = ap.parse_args()

    out = Path(args.out_dir)
    panel_dir = out / "panels"
    metrics_dir = out / "metrics"
    panel_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in collect(args.input_dir):
        ff = rgb_hf(s["fugr"], s["cn"], args.beta, args.sigma)
        d = psnr(ff, s["gt"]) - psnr(s["fugr"], s["gt"])
        rows.append((d, s, ff))

    rows.sort(key=lambda x: x[0], reverse=True)
    chosen = rows[:args.panel_count // 2] + rows[-(args.panel_count - args.panel_count // 2):]

    txt = metrics_dir / "B19_zoom_panel_selection.txt"
    with txt.open("w") as f:
        f.write("B19 qualitative zoom panel selection\n")
        f.write(f"beta={args.beta}, sigma={args.sigma}\n\n")
        for rank, (d, s, ff) in enumerate(chosen):
            box = crop_around_edge(s["gt"], args.crop_size)
            name = f"B19_zoom_rank{rank:02d}_seq{s['seq']}_{s['frame']}_dpsnr{d:+.4f}.png"
            panel(
                panel_dir / name,
                [
                    crop(s["basic"], box),
                    crop(s["fugr"], box),
                    crop(ff, box),
                    crop(s["cn"], box),
                    crop(s["gt"], box),
                    crop(err_map(s["fugr"], s["gt"]), box),
                    crop(err_map(ff, s["gt"]), box),
                ],
                ["Basic", "FUGR-C", "FreqFusion", "ControlNet", "GT", "FUGR Err", "Fusion Err"],
            )
            f.write(f"{rank},{s['seq']},{s['frame']},{d:.8f},{name}\n")

    print(txt.read_text())
    print("Saved panels:", panel_dir)

if __name__ == "__main__":
    main()
