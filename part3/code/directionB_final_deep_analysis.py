#!/usr/bin/env python3
"""
Direction B final deep analysis.

This script strengthens Direction B without rerunning diffusion:
B16. Edge/high-frequency region analysis
B17. Leave-one-sequence-out parameter generalization
B18. Parameter sensitivity heatmap
B19. Qualitative zoom-in panels

Expected input:
  expanded_strong_st015/frames/<seq>/<frame>_basic.png
  expanded_strong_st015/frames/<seq>/<frame>_fugr.png
  expanded_strong_st015/frames/<seq>/<frame>_controlnet_fugr.png
  expanded_strong_st015/frames/<seq>/<frame>_gt.png
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


def save_rgb(p, img):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(img * 255, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(p), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def gray(img):
    return cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def rgb_hf(fugr, cn, beta, sigma):
    return np.clip(fugr + beta * (highpass(cn, sigma) - highpass(fugr, sigma)), 0, 1)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def masked_psnr(a, b, mask):
    if mask.sum() < 8:
        return float("nan")
    diff = (a - b) ** 2
    mse = float(np.mean(diff[mask]))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def ssim_rgb(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        x, y = a[..., ch].astype(np.float32), b[..., ch].astype(np.float32)
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


def edge_mask_from_gt(gt, percentile=80.0):
    g = gray(gt)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, percentile)
    mask = mag >= thr
    return mask


def hf_mae(a, b, sigma):
    return float(np.mean(np.abs(highpass(a, sigma) - highpass(b, sigma))))


def colorize(x):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    y = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    cm = cv2.applyColorMap((y * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def err_map(a, b):
    return colorize(np.mean(np.abs(a - b), axis=2))


def resize_h(img, h=210):
    H, W = img.shape[:2]
    return cv2.resize(img, (int(W * h / H), h), interpolation=cv2.INTER_AREA)


def crop_around_edge(gt, crop=160):
    g = gray(gt)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 4.0)
    h, w = mag.shape
    y, x = np.unravel_index(np.argmax(mag), mag.shape)
    half = crop // 2
    x0 = max(0, min(w - crop, x - half))
    y0 = max(0, min(h - crop, y - half))
    return int(x0), int(y0), int(crop), int(crop)


def crop(img, box):
    x, y, w, h = box
    return img[y:y+h, x:x+w]


def make_panel(path, imgs, titles):
    imgs = [resize_h(x, 210) for x in imgs]
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
    data = {}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        items = []
        for fugr_path in sorted(sd.glob("*_fugr.png")):
            frame = fugr_path.name.replace("_fugr.png", "")
            basic_path = sd / f"{frame}_basic.png"
            cn_path = sd / f"{frame}_controlnet_fugr.png"
            gt_path = sd / f"{frame}_gt.png"
            if cn_path.exists() and gt_path.exists():
                items.append({
                    "frame": frame,
                    "basic": read_rgb(basic_path) if basic_path.exists() else read_rgb(fugr_path),
                    "fugr": read_rgb(fugr_path),
                    "cn": read_rgb(cn_path),
                    "gt": read_rgb(gt_path),
                })
        if items:
            data[sd.name] = items
    return data


def eval_method(data, seqs, method, make_img, edge_percentile, hp_sigma):
    frame_rows = []
    seq_rows = []
    for seq in seqs:
        outs, gts = [], []
        for it in data[seq]:
            out = make_img(it)
            gt = it["gt"]
            m_edge = edge_mask_from_gt(gt, edge_percentile)
            m_flat = ~m_edge
            outs.append(out)
            gts.append(gt)
            frame_rows.append({
                "sequence": seq,
                "frame": it["frame"],
                "method": method,
                "psnr": psnr(out, gt),
                "ssim": ssim_rgb(out, gt),
                "sharpness": sharpness(out),
                "edge_psnr": masked_psnr(out, gt, m_edge),
                "flat_psnr": masked_psnr(out, gt, m_flat),
                "hf_mae": hf_mae(out, gt, hp_sigma),
            })
        seq_rows.append({
            "sequence": seq,
            "method": method,
            "num_frames": len(outs),
            "psnr": float(np.mean([r["psnr"] for r in frame_rows if r["sequence"] == seq and r["method"] == method])),
            "ssim": float(np.mean([r["ssim"] for r in frame_rows if r["sequence"] == seq and r["method"] == method])),
            "sharpness": float(np.mean([r["sharpness"] for r in frame_rows if r["sequence"] == seq and r["method"] == method])),
            "edge_psnr": float(np.mean([r["edge_psnr"] for r in frame_rows if r["sequence"] == seq and r["method"] == method])),
            "flat_psnr": float(np.mean([r["flat_psnr"] for r in frame_rows if r["sequence"] == seq and r["method"] == method])),
            "hf_mae": float(np.mean([r["hf_mae"] for r in frame_rows if r["sequence"] == seq and r["method"] == method])),
            "tde": tde(outs, gts),
        })
    return frame_rows, seq_rows


def aggregate(seq_rows, split):
    return {
        "split": split,
        "method": seq_rows[0]["method"],
        "num_sequences": len(seq_rows),
        "num_frames": int(sum(r["num_frames"] for r in seq_rows)),
        "psnr": float(np.mean([r["psnr"] for r in seq_rows])),
        "ssim": float(np.mean([r["ssim"] for r in seq_rows])),
        "sharpness": float(np.mean([r["sharpness"] for r in seq_rows])),
        "edge_psnr": float(np.mean([r["edge_psnr"] for r in seq_rows])),
        "flat_psnr": float(np.mean([r["flat_psnr"] for r in seq_rows])),
        "hf_mae": float(np.mean([r["hf_mae"] for r in seq_rows])),
        "tde": float(np.mean([r["tde"] for r in seq_rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_seqs", nargs="+", default=["000", "003", "006", "010", "011"])
    ap.add_argument("--test_seqs", nargs="+", default=["018", "020", "026", "028", "029"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.8])
    ap.add_argument("--selected_beta", type=float, default=0.60)
    ap.add_argument("--selected_sigma", type=float, default=0.4)
    ap.add_argument("--expanded_beta", type=float, default=0.30)
    ap.add_argument("--expanded_sigma", type=float, default=0.5)
    ap.add_argument("--edge_percentile", type=float, default=80.0)
    ap.add_argument("--hp_sigma", type=float, default=1.0)
    ap.add_argument("--panel_count", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    fig_dir = out_dir / "figures"
    panel_dir = out_dir / "panels"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    data = collect(args.input_dir)
    all_seqs = args.calib_seqs + args.test_seqs
    missing = [s for s in all_seqs if s not in data]
    if missing:
        raise FileNotFoundError(f"Missing sequences: {missing}")

    print("Loaded sequences:", sorted(data.keys()), flush=True)

    # B16: local edge/HF metrics for key methods.
    key_methods = [
        ("BasicVSR", lambda it: it["basic"]),
        ("FUGR-C", lambda it: it["fugr"]),
        ("ControlNet-FUGR", lambda it: it["cn"]),
        (f"FreqFusion-selected-b{args.selected_beta:.2f}-s{args.selected_sigma:.1f}",
         lambda it: rgb_hf(it["fugr"], it["cn"], args.selected_beta, args.selected_sigma)),
        (f"FreqFusion-expandedbest-b{args.expanded_beta:.2f}-s{args.expanded_sigma:.1f}",
         lambda it: rgb_hf(it["fugr"], it["cn"], args.expanded_beta, args.expanded_sigma)),
    ]

    all_frame_rows, all_seq_rows, local_summary = [], [], []
    for split, seqs in [("all40", all_seqs), ("calibration", args.calib_seqs), ("test", args.test_seqs)]:
        print("B16 evaluating split:", split, flush=True)
        for method, fn in key_methods:
            fr, sr = eval_method(data, seqs, method, fn, args.edge_percentile, args.hp_sigma)
            all_frame_rows.extend([{"split": split, **r} for r in fr])
            all_seq_rows.extend([{"split": split, **r} for r in sr])
            local_summary.append(aggregate(sr, split))

    # Deltas versus FUGR-C within each split.
    fugr_by_split = {r["split"]: r for r in local_summary if r["method"] == "FUGR-C"}
    for r in local_summary:
        base = fugr_by_split[r["split"]]
        for k in ["psnr", "ssim", "sharpness", "edge_psnr", "flat_psnr", "hf_mae", "tde"]:
            r[f"delta_{k}_vs_fugr"] = r[k] - base[k]

    with (metrics_dir / "B16_edge_hf_summary.csv").open("w", newline="") as f:
        fields = ["split", "method", "num_sequences", "num_frames", "psnr", "ssim", "sharpness",
                  "edge_psnr", "flat_psnr", "hf_mae", "tde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_sharpness_vs_fugr",
                  "delta_edge_psnr_vs_fugr", "delta_flat_psnr_vs_fugr",
                  "delta_hf_mae_vs_fugr", "delta_tde_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(local_summary)

    with (metrics_dir / "B16_edge_hf_sequence_metrics.csv").open("w", newline="") as f:
        fields = ["split", "sequence", "method", "num_frames", "psnr", "ssim", "sharpness",
                  "edge_psnr", "flat_psnr", "hf_mae", "tde"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_seq_rows)

    with (metrics_dir / "B16_edge_hf_frame_metrics.csv").open("w", newline="") as f:
        fields = ["split", "sequence", "frame", "method", "psnr", "ssim", "sharpness",
                  "edge_psnr", "flat_psnr", "hf_mae"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_frame_rows)

    # B17: leave-one-sequence-out parameter generalization.
    print("B17 leave-one-sequence-out validation", flush=True)
    loso_rows = []
    for held in all_seqs:
        train = [s for s in all_seqs if s != held]
        config_scores = []
        for beta in args.betas:
            for sigma in args.sigmas:
                method = f"RGB-HF-b{beta:.2f}-s{sigma:.1f}"
                fn = lambda it, b=beta, s=sigma: rgb_hf(it["fugr"], it["cn"], b, s)
                _, sr_train = eval_method(data, train, method, fn, args.edge_percentile, args.hp_sigma)
                train_sum = aggregate(sr_train, "train")
                config_scores.append({"beta": beta, "sigma": sigma, **train_sum})
        best = sorted(config_scores, key=lambda r: (r["psnr"], r["ssim"], -r["tde"]), reverse=True)[0]
        beta, sigma = best["beta"], best["sigma"]

        method = f"LOSO-selected-b{beta:.2f}-s{sigma:.1f}"
        fn = lambda it, b=beta, s=sigma: rgb_hf(it["fugr"], it["cn"], b, s)
        _, sr_test = eval_method(data, [held], method, fn, args.edge_percentile, args.hp_sigma)
        test_sum = aggregate(sr_test, "heldout")

        _, sr_fugr = eval_method(data, [held], "FUGR-C", lambda it: it["fugr"], args.edge_percentile, args.hp_sigma)
        fugr_sum = aggregate(sr_fugr, "heldout")

        loso_rows.append({
            "heldout_sequence": held,
            "selected_beta": beta,
            "selected_sigma": sigma,
            "train_psnr": best["psnr"],
            "heldout_psnr": test_sum["psnr"],
            "heldout_ssim": test_sum["ssim"],
            "heldout_edge_psnr": test_sum["edge_psnr"],
            "heldout_hf_mae": test_sum["hf_mae"],
            "heldout_tde": test_sum["tde"],
            "fugr_psnr": fugr_sum["psnr"],
            "fugr_ssim": fugr_sum["ssim"],
            "fugr_edge_psnr": fugr_sum["edge_psnr"],
            "fugr_hf_mae": fugr_sum["hf_mae"],
            "fugr_tde": fugr_sum["tde"],
            "delta_psnr_vs_fugr": test_sum["psnr"] - fugr_sum["psnr"],
            "delta_ssim_vs_fugr": test_sum["ssim"] - fugr_sum["ssim"],
            "delta_edge_psnr_vs_fugr": test_sum["edge_psnr"] - fugr_sum["edge_psnr"],
            "delta_hf_mae_vs_fugr": test_sum["hf_mae"] - fugr_sum["hf_mae"],
            "delta_tde_vs_fugr": test_sum["tde"] - fugr_sum["tde"],
        })

    with (metrics_dir / "B17_loso_generalization.csv").open("w", newline="") as f:
        fields = ["heldout_sequence", "selected_beta", "selected_sigma", "train_psnr",
                  "heldout_psnr", "heldout_ssim", "heldout_edge_psnr", "heldout_hf_mae", "heldout_tde",
                  "fugr_psnr", "fugr_ssim", "fugr_edge_psnr", "fugr_hf_mae", "fugr_tde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_edge_psnr_vs_fugr",
                  "delta_hf_mae_vs_fugr", "delta_tde_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(loso_rows)

    # B18: all40 config sensitivity.
    print("B18 all40 parameter sensitivity", flush=True)
    grid_rows = []
    for beta in args.betas:
        for sigma in args.sigmas:
            method = f"RGB-HF-b{beta:.2f}-s{sigma:.1f}"
            fn = lambda it, b=beta, s=sigma: rgb_hf(it["fugr"], it["cn"], b, s)
            _, sr = eval_method(data, all_seqs, method, fn, args.edge_percentile, args.hp_sigma)
            rec = aggregate(sr, "all40")
            rec["beta"] = beta
            rec["sigma"] = sigma
            grid_rows.append(rec)

    fugr_all = next(r for r in local_summary if r["split"] == "all40" and r["method"] == "FUGR-C")
    for r in grid_rows:
        r["delta_psnr_vs_fugr"] = r["psnr"] - fugr_all["psnr"]
        r["delta_ssim_vs_fugr"] = r["ssim"] - fugr_all["ssim"]
        r["delta_edge_psnr_vs_fugr"] = r["edge_psnr"] - fugr_all["edge_psnr"]
        r["delta_tde_vs_fugr"] = r["tde"] - fugr_all["tde"]

    grid_rows_sorted = sorted(grid_rows, key=lambda r: (r["psnr"], r["ssim"], -r["tde"]), reverse=True)

    with (metrics_dir / "B18_config_grid_all40.csv").open("w", newline="") as f:
        fields = ["split", "method", "num_sequences", "num_frames", "psnr", "ssim", "sharpness",
                  "edge_psnr", "flat_psnr", "hf_mae", "tde", "beta", "sigma",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr",
                  "delta_edge_psnr_vs_fugr", "delta_tde_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(grid_rows_sorted)

    # Heatmaps.
    betas = args.betas
    sigmas = args.sigmas
    mat_psnr = np.zeros((len(betas), len(sigmas)), dtype=np.float32)
    mat_edge = np.zeros_like(mat_psnr)
    mat_tde = np.zeros_like(mat_psnr)
    for i, b in enumerate(betas):
        for j, s in enumerate(sigmas):
            r = next(x for x in grid_rows if abs(x["beta"] - b) < 1e-9 and abs(x["sigma"] - s) < 1e-9)
            mat_psnr[i, j] = r["delta_psnr_vs_fugr"]
            mat_edge[i, j] = r["delta_edge_psnr_vs_fugr"]
            mat_tde[i, j] = r["delta_tde_vs_fugr"]

    def plot_heat(mat, title, name, cmap="viridis"):
        plt.figure(figsize=(6, 4))
        plt.imshow(mat, aspect="auto", cmap=cmap)
        plt.colorbar()
        plt.xticks(range(len(sigmas)), [str(x) for x in sigmas])
        plt.yticks(range(len(betas)), [str(x) for x in betas])
        plt.xlabel("High-pass sigma")
        plt.ylabel("Beta")
        plt.title(title)
        for i in range(len(betas)):
            for j in range(len(sigmas)):
                plt.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=300)
        plt.close()

    plot_heat(mat_psnr, "B18 all40: PSNR delta vs FUGR-C", "B18_heatmap_delta_psnr.png")
    plot_heat(mat_edge, "B18 all40: Edge-PSNR delta vs FUGR-C", "B18_heatmap_delta_edge_psnr.png")
    plot_heat(mat_tde, "B18 all40: TDE delta vs FUGR-C", "B18_heatmap_delta_tde.png", cmap="coolwarm")

    # B19 qualitative panels: top positive and negative frame deltas for selected config.
    print("B19 qualitative zoom panels", flush=True)
    selected_name = f"FreqFusion-selected-b{args.selected_beta:.2f}-s{args.selected_sigma:.1f}"
    frame_delta_rows = []
    for seq in all_seqs:
        for it in data[seq]:
            ff = rgb_hf(it["fugr"], it["cn"], args.selected_beta, args.selected_sigma)
            d = psnr(ff, it["gt"]) - psnr(it["fugr"], it["gt"])
            frame_delta_rows.append((d, seq, it["frame"], it, ff))

    frame_delta_rows.sort(key=lambda x: x[0], reverse=True)
    chosen = frame_delta_rows[: args.panel_count // 2] + frame_delta_rows[-(args.panel_count - args.panel_count // 2):]
    for rank, (d, seq, frame, it, ff) in enumerate(chosen):
        box = crop_around_edge(it["gt"], crop=160)
        panel(
            panel_dir / f"B19_zoom_rank{rank:02d}_seq{seq}_{frame}_dpsnr{d:+.4f}.png",
            [
                crop(it["fugr"], box),
                crop(ff, box),
                crop(it["cn"], box),
                crop(it["gt"], box),
                crop(err_map(it["fugr"], it["gt"]), box),
                crop(err_map(ff, it["gt"]), box),
            ],
            ["FUGR-C", "FreqFusion", "ControlNet", "GT", "FUGR Err", "Fusion Err"],
        )

    # Final readme.
    loso_deltas = np.array([r["delta_psnr_vs_fugr"] for r in loso_rows], dtype=np.float64)
    loso_edge = np.array([r["delta_edge_psnr_vs_fugr"] for r in loso_rows], dtype=np.float64)
    loso_tde = np.array([r["delta_tde_vs_fugr"] for r in loso_rows], dtype=np.float64)
    selected_test = next(r for r in local_summary if r["split"] == "test" and r["method"].startswith("FreqFusion-selected"))
    selected_all = next(r for r in local_summary if r["split"] == "all40" and r["method"].startswith("FreqFusion-selected"))

    txt = metrics_dir / "B_final_deep_analysis_summary.txt"
    with txt.open("w") as f:
        f.write("Direction B Final Deep Analysis\n\n")
        f.write("B16 Edge/High-frequency analysis\n")
        f.write(str(selected_all) + "\n\n")
        f.write("B17 Leave-one-sequence-out generalization\n")
        f.write(f"mean LOSO delta PSNR: {float(np.mean(loso_deltas)):.8f}\n")
        f.write(f"positive LOSO sequences: {int(np.sum(loso_deltas > 0))}/{len(loso_deltas)}\n")
        f.write(f"mean LOSO delta edge PSNR: {float(np.mean(loso_edge)):.8f}\n")
        f.write(f"mean LOSO delta TDE: {float(np.mean(loso_tde)):.8f}\n\n")
        f.write("B15-style selected config on test split, recomputed with edge/HF metrics\n")
        f.write(str(selected_test) + "\n\n")
        f.write("Best all40 config from B18\n")
        f.write(str(grid_rows_sorted[0]) + "\n\n")
        f.write("Interpretation\n")
        f.write(
            "Direct ControlNet output is unsafe, but frequency-constrained residual fusion "
            "has a small fidelity gain. This script checks whether that gain appears in "
            "edge/high-frequency metrics, whether parameter choice is robust under LOSO validation, "
            "and where qualitative improvements/failures occur.\n"
        )

    print(txt.read_text())
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
