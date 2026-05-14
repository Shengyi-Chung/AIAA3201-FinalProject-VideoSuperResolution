import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def normalize01(x, eps=1e-8):
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def to_gray(img):
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def texture_map(img):
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = cv2.GaussianBlur(grad, (0, 0), 1.2)
    return normalize01(grad)


def disagreement_map(basic, gan):
    diff = np.mean(np.abs(basic - gan), axis=2)
    diff = cv2.GaussianBlur(diff, (0, 0), 1.5)
    return diff


def compute_alpha(basic, gan, max_alpha=0.28, tau=0.08):
    tex = texture_map(basic)
    disagree = disagreement_map(basic, gan)

    reliability = np.exp(-disagree / tau)

    alpha = max_alpha * tex * reliability
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
    alpha = np.clip(alpha, 0.0, max_alpha)

    return alpha, tex, disagree, reliability


def hybrid_frame(basic, gan, alpha):
    a = alpha[..., None]
    out = (1.0 - a) * basic + a * gan
    return np.clip(out, 0.0, 1.0)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def ssim_rgb(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    scores = []

    for c in range(3):
        x = img1[..., c].astype(np.float32)
        y = img2[..., c].astype(np.float32)

        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
        scores.append(float(np.mean(num / (den + 1e-12))))

    return float(np.mean(scores))


def colorize(x):
    x = normalize01(x)
    u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out.astype(np.float32) / 255.0


def make_panel(save_path, imgs, titles):
    h, w = imgs[0].shape[:2]
    title_h = 38
    panel = np.ones((h + title_h, w * len(imgs), 3), dtype=np.uint8) * 255

    for i, (img, title) in enumerate(zip(imgs, titles)):
        x0 = i * w
        img_u8 = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
        panel[title_h:title_h + h, x0:x0 + w] = img_u8

        cv2.putText(
            panel,
            title,
            (x0 + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), panel)


def find_pairs(basic_dir, gan_dir):
    basic_dir = Path(basic_dir)
    gan_dir = Path(gan_dir)

    pairs = []
    for bp in sorted(basic_dir.rglob("*.png")):
        rel = bp.relative_to(basic_dir)
        gp = gan_dir / rel
        if gp.exists():
            pairs.append((rel, bp, gp))

    if len(pairs) == 0:
        raise RuntimeError("No matched frames found.")

    return pairs


def find_gt(gt_dir, rel):
    if gt_dir is None:
        return None

    gt_dir = Path(gt_dir)

    candidates = [
        gt_dir / rel,
        gt_dir / rel.name,
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = list(gt_dir.rglob(rel.name))
    if len(matches) > 0:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic_dir", default="basicvsr_result")
    parser.add_argument("--gan_dir", default="vsrgan_result")
    parser.add_argument("--out_dir", default="part3_result")
    parser.add_argument("--fig_dir", default="figures")
    parser.add_argument("--csv_path", default="evalresult/part3_hybrid_metrics.csv")
    parser.add_argument("--summary_path", default="evalresult/part3_summary.txt")
    parser.add_argument("--gt_dir", default=None)
    parser.add_argument("--max_alpha", type=float, default=0.28)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--temporal_smooth", type=float, default=0.65)
    parser.add_argument("--panel_every", type=int, default=25)
    args = parser.parse_args()

    pairs = find_pairs(args.basic_dir, args.gan_dir)
    print(f"Matched frames: {len(pairs)}")

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    prev_alpha = None
    gt_count = 0

    for idx, (rel, bp, gp) in enumerate(pairs):
        basic = read_rgb(bp)
        gan = read_rgb(gp)

        alpha, tex, disagree, reliability = compute_alpha(
            basic,
            gan,
            max_alpha=args.max_alpha,
            tau=args.tau,
        )

        if prev_alpha is not None and prev_alpha.shape == alpha.shape:
            alpha = args.temporal_smooth * prev_alpha + (1.0 - args.temporal_smooth) * alpha

        prev_alpha = alpha.copy()

        hybrid = hybrid_frame(basic, gan, alpha)
        save_rgb(out_dir / rel, hybrid)

        gt_path = find_gt(args.gt_dir, rel) if args.gt_dir else None
        cur_psnr = ""
        cur_ssim = ""

        gt_img = None
        if gt_path is not None:
            gt_img = read_rgb(gt_path)
            if gt_img.shape[:2] != hybrid.shape[:2]:
                gt_img = cv2.resize(gt_img, (hybrid.shape[1], hybrid.shape[0]), interpolation=cv2.INTER_CUBIC)
            cur_psnr = psnr(hybrid, gt_img)
            cur_ssim = ssim_rgb(hybrid, gt_img)
            gt_count += 1

        rows.append([
            str(rel),
            cur_psnr,
            cur_ssim,
            float(alpha.mean()),
            float(alpha.max()),
        ])

        if idx % args.panel_every == 0:
            imgs = [
                basic,
                gan,
                hybrid,
                colorize(alpha),
                colorize(disagree),
            ]
            titles = [
                "BasicVSR",
                "VSRGAN",
                "U-Hybrid",
                "Alpha",
                "Disagreement",
            ]

            if gt_img is not None:
                imgs.append(gt_img)
                titles.append("GT")

            panel_name = "panel_" + str(rel).replace("/", "_")
            make_panel(fig_dir / panel_name, imgs, titles)

    with open(args.csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "psnr", "ssim", "alpha_mean", "alpha_max"])
        for r in rows:
            writer.writerow(r)

        valid = [(float(r[1]), float(r[2])) for r in rows if r[1] != "" and r[2] != ""]
        if len(valid) > 0:
            avg_psnr = np.mean([x[0] for x in valid])
            avg_ssim = np.mean([x[1] for x in valid])
            writer.writerow(["average", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", "", ""])

    with open(args.summary_path, "w") as f:
        f.write("Part3: Uncertainty-Aware Hybrid VSR\n")
        f.write(f"Matched frames: {len(pairs)}\n")
        f.write(f"GT frames found: {gt_count}\n")
        f.write(f"max_alpha: {args.max_alpha}\n")
        f.write(f"tau: {args.tau}\n")
        f.write(f"temporal_smooth: {args.temporal_smooth}\n")

        valid = [(float(r[1]), float(r[2])) for r in rows if r[1] != "" and r[2] != ""]
        if len(valid) > 0:
            avg_psnr = np.mean([x[0] for x in valid])
            avg_ssim = np.mean([x[1] for x in valid])
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Saved hybrid frames to: {args.out_dir}")
    print(f"Saved visual panels to: {args.fig_dir}")
    print(f"Saved metrics to: {args.csv_path}")
    print(f"Saved summary to: {args.summary_path}")


if __name__ == "__main__":
    main()
