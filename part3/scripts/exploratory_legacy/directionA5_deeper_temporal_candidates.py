#!/usr/bin/env python3
"""
Direction A5: Deeper temporal-candidate and metric-robustness analysis.

Why this exists:
  A1-A4 mainly tested temporal smoothing / residual smoothing / risk gating / lambda headroom.
  A5 adds two missing checks:
    1) metric robustness: raw TDE and flow-warped TDE (WTDE);
    2) additional post-hoc temporal candidates:
       - residual median stabilization,
       - residual attenuation by local temporal inconsistency,
       - residual temporal clipping.

Input format:
  input_dir/frames/<seq>/<frame>_basic.png
  input_dir/frames/<seq>/<frame>_fugr.png
  input_dir/frames/<seq>/<frame>_gt.png

Outputs:
  metrics/A5_summary_metrics.csv
  metrics/A5_sequence_metrics.csv
  metrics/A5_headroom_summary.txt
  figures/A5_pareto_psnr_tde.png
  figures/A5_pareto_psnr_wtde.png
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def gray(img):
    return cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def ssim_rgb(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        x = a[..., ch].astype(np.float32)
        y = b[..., ch].astype(np.float32)
        mx = cv2.GaussianBlur(x, (11, 11), 1.5)
        my = cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
        vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
        vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
        vals.append(float(np.mean(((2 * mx * my + c1) * (2 * vxy + c2)) /
                                  ((mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12))))
    return float(np.mean(vals))


def sharpness(img):
    return float(np.var(cv2.Laplacian(gray(img), cv2.CV_32F)))


def tde(xs, ys):
    if len(xs) < 2:
        return 0.0
    vals = []
    for i in range(1, len(xs)):
        vals.append(float(np.mean(np.abs((xs[i] - xs[i - 1]) - (ys[i] - ys[i - 1])))))
    return float(np.mean(vals))


def flow_cur_to_prev(cur, prev):
    # Flow from current image coordinates to previous image.
    return cv2.calcOpticalFlowFarneback(
        gray(cur), gray(prev), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )


def warp_prev_to_cur(prev_img, flow_cur_to_prev_):
    h, w = flow_cur_to_prev_.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    mx = (xx + flow_cur_to_prev_[..., 0]).astype(np.float32)
    my = (yy + flow_cur_to_prev_[..., 1]).astype(np.float32)
    return cv2.remap(prev_img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def wtde(xs, ys):
    """Flow-warped temporal difference error, using GT-derived optical flow for metric only."""
    if len(xs) < 2:
        return 0.0
    vals = []
    for i in range(1, len(xs)):
        fl = flow_cur_to_prev(ys[i], ys[i - 1])
        wp = warp_prev_to_cur(xs[i - 1], fl)
        wg = warp_prev_to_cur(ys[i - 1], fl)
        vals.append(float(np.mean(np.abs((xs[i] - wp) - (ys[i] - wg)))))
    return float(np.mean(vals))


def motion(ys):
    if len(ys) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(ys[i] - ys[i - 1])) for i in range(1, len(ys))]))


def collect(input_dir):
    root = Path(input_dir) / "frames"
    data = {}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        items = []
        for fugr_path in sorted(sd.glob("*_fugr.png")):
            frame = fugr_path.name.replace("_fugr.png", "")
            basic_path = sd / f"{frame}_basic.png"
            gt_path = sd / f"{frame}_gt.png"
            if basic_path.exists() and gt_path.exists():
                items.append({
                    "frame": frame,
                    "basic": read_rgb(basic_path),
                    "fugr": read_rgb(fugr_path),
                    "gt": read_rgb(gt_path),
                })
        if items:
            data[sd.name] = items
    return data


def residuals(basic, fugr):
    return [f - b for b, f in zip(basic, fugr)]


def make_lambda(basic, fugr, lam):
    return [np.clip(b + lam * (f - b), 0, 1) for b, f in zip(basic, fugr)]


def make_residual_median(basic, fugr, beta):
    r = residuals(basic, fugr)
    outs = []
    for i in range(len(r)):
        neigh = [r[i]]
        if i > 0:
            neigh.append(r[i - 1])
        if i + 1 < len(r):
            neigh.append(r[i + 1])
        med = np.median(np.stack(neigh, axis=0), axis=0)
        rr = (1 - beta) * r[i] + beta * med
        outs.append(np.clip(basic[i] + rr, 0, 1))
    return outs


def make_residual_attenuation(basic, fugr, tau, gamma_min, blur_sigma):
    r = residuals(basic, fugr)
    outs = []
    for i in range(len(r)):
        neigh = []
        if i > 0:
            neigh.append(r[i - 1])
        if i + 1 < len(r):
            neigh.append(r[i + 1])
        if not neigh:
            gamma = np.ones(r[i].shape[:2], dtype=np.float32)
        else:
            mean_neigh = np.mean(np.stack(neigh, axis=0), axis=0)
            risk = np.mean(np.abs(r[i] - mean_neigh), axis=2)
            gamma = gamma_min + (1 - gamma_min) * np.exp(-risk / tau)
            if blur_sigma > 0:
                gamma = cv2.GaussianBlur(gamma, (0, 0), blur_sigma)
            gamma = np.clip(gamma, gamma_min, 1.0)
        outs.append(np.clip(basic[i] + gamma[..., None] * r[i], 0, 1))
    return outs


def make_residual_clip(basic, fugr, clip_scale):
    """Clip residual magnitude by a robust local temporal residual magnitude."""
    r = residuals(basic, fugr)
    outs = []
    eps = 1e-8
    for i in range(len(r)):
        neigh = [np.abs(r[i])]
        if i > 0:
            neigh.append(np.abs(r[i - 1]))
        if i + 1 < len(r):
            neigh.append(np.abs(r[i + 1]))
        ref_mag = np.median(np.stack(neigh, axis=0), axis=0)
        limit = clip_scale * ref_mag + 1e-4
        rr = np.clip(r[i], -limit, limit)
        outs.append(np.clip(basic[i] + rr, 0, 1))
    return outs


def eval_seq(xs, ys):
    return {
        "num_frames": len(xs),
        "motion": motion(ys),
        "psnr": float(np.mean([psnr(x, y) for x, y in zip(xs, ys)])),
        "ssim": float(np.mean([ssim_rgb(x, y) for x, y in zip(xs, ys)])),
        "sharpness": float(np.mean([sharpness(x) for x in xs])),
        "tde": tde(xs, ys),
        "wtde": wtde(xs, ys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2])
    ap.add_argument("--median_betas", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.00])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.002, 0.005, 0.010, 0.020])
    ap.add_argument("--gamma_mins", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--clip_scales", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.5])
    ap.add_argument("--blur_sigma", type=float, default=0.8)
    ap.add_argument("--psnr_loss_limit", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    fig_dir = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = collect(args.input_dir)
    print("Loaded sequences:", sorted(data.keys()), flush=True)

    method_builders = []
    method_builders.append(("BasicVSR", lambda b, f: b))
    method_builders.append(("FUGR-C", lambda b, f: f))

    for lam in args.lambdas:
        method_builders.append((f"Lambda-{lam:.2f}", lambda b, f, lam=lam: make_lambda(b, f, lam)))

    for beta in args.median_betas:
        method_builders.append((f"ResidualMedian-b{beta:.2f}", lambda b, f, beta=beta: make_residual_median(b, f, beta)))

    for tau in args.taus:
        for gmin in args.gamma_mins:
            method_builders.append((f"ResidualAtten-t{tau:.3f}-g{gmin:.2f}",
                                    lambda b, f, tau=tau, gmin=gmin: make_residual_attenuation(b, f, tau, gmin, args.blur_sigma)))

    for cs in args.clip_scales:
        method_builders.append((f"ResidualClip-c{cs:.2f}", lambda b, f, cs=cs: make_residual_clip(b, f, cs)))

    seq_rows = []
    for method, builder in method_builders:
        print("Evaluating", method, flush=True)
        for seq, items in data.items():
            basic = [it["basic"] for it in items]
            fugr = [it["fugr"] for it in items]
            gt = [it["gt"] for it in items]
            outs = builder(basic, fugr)
            rec = eval_seq(outs, gt)
            rec.update({"sequence": seq, "method": method})
            seq_rows.append(rec)

    methods = sorted(set(r["method"] for r in seq_rows))
    summary = []
    for method in methods:
        rows = [r for r in seq_rows if r["method"] == method]
        rec = {
            "method": method,
            "num_sequences": len(rows),
            "num_frames": int(sum(r["num_frames"] for r in rows)),
            "motion": float(np.mean([r["motion"] for r in rows])),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "sharpness": float(np.mean([r["sharpness"] for r in rows])),
            "tde": float(np.mean([r["tde"] for r in rows])),
            "wtde": float(np.mean([r["wtde"] for r in rows])),
        }
        summary.append(rec)

    fugr = next(r for r in summary if r["method"] == "FUGR-C")
    for r in summary:
        r["delta_psnr_vs_fugr"] = r["psnr"] - fugr["psnr"]
        r["delta_ssim_vs_fugr"] = r["ssim"] - fugr["ssim"]
        r["delta_tde_vs_fugr"] = r["tde"] - fugr["tde"]
        r["delta_wtde_vs_fugr"] = r["wtde"] - fugr["wtde"]
        r["tde_reduction_pct_vs_fugr"] = 100 * (fugr["tde"] - r["tde"]) / fugr["tde"] if fugr["tde"] > 0 else 0
        r["wtde_reduction_pct_vs_fugr"] = 100 * (fugr["wtde"] - r["wtde"]) / fugr["wtde"] if fugr["wtde"] > 0 else 0

    summary_by_tde = sorted(summary, key=lambda r: (r["tde"], -r["psnr"]))
    summary_by_wtde = sorted(summary, key=lambda r: (r["wtde"], -r["psnr"]))
    summary_by_psnr = sorted(summary, key=lambda r: (r["psnr"], -r["tde"]), reverse=True)

    with (metrics_dir / "A5_sequence_metrics.csv").open("w", newline="") as f:
        fields = ["sequence", "method", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde", "wtde"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(seq_rows)

    with (metrics_dir / "A5_summary_metrics.csv").open("w", newline="") as f:
        fields = ["method", "num_sequences", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde", "wtde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_tde_vs_fugr", "delta_wtde_vs_fugr",
                  "tde_reduction_pct_vs_fugr", "wtde_reduction_pct_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_by_tde)

    def scatter(metric, ylabel, name):
        plt.figure(figsize=(7, 5))
        xs = [r[metric] for r in summary]
        ys = [r["psnr"] for r in summary]
        plt.scatter(xs, ys, s=22)
        plt.scatter([fugr[metric]], [fugr["psnr"]], s=90, marker="*")
        plt.xlabel(f"{metric.upper()} lower is better")
        plt.ylabel("PSNR higher is better")
        plt.title(f"A5 PSNR-{metric.upper()} tradeoff")
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=300)
        plt.close()

    scatter("tde", "TDE", "A5_pareto_psnr_tde.png")
    scatter("wtde", "WTDE", "A5_pareto_psnr_wtde.png")

    candidates_tde = [r for r in summary if r["delta_psnr_vs_fugr"] >= -args.psnr_loss_limit]
    best_tde_under_psnr = sorted(candidates_tde, key=lambda r: (r["tde"], -r["psnr"]))[0]
    best_wtde_under_psnr = sorted(candidates_tde, key=lambda r: (r["wtde"], -r["psnr"]))[0]

    txt = metrics_dir / "A5_headroom_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A5: Deeper Temporal Candidate and Metric-Robustness Analysis\n\n")
        f.write("FUGR-C reference:\n")
        f.write(str(fugr) + "\n\n")
        f.write("Best by raw TDE under PSNR-loss constraint:\n")
        f.write(str(best_tde_under_psnr) + "\n\n")
        f.write("Best by flow-warped TDE under PSNR-loss constraint:\n")
        f.write(str(best_wtde_under_psnr) + "\n\n")
        f.write("Top 20 by raw TDE:\n")
        for r in summary_by_tde[:20]:
            f.write(str(r) + "\n")
        f.write("\nTop 20 by WTDE:\n")
        for r in summary_by_wtde[:20]:
            f.write(str(r) + "\n")
        f.write("\nTop 10 by PSNR:\n")
        for r in summary_by_psnr[:10]:
            f.write(str(r) + "\n")

    print(txt.read_text())
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
