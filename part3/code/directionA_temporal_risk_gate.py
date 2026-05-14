#!/usr/bin/env python3
"""
Direction A3: Temporal-Risk Residual Gate (TRG)

A1/A2 showed that temporal averaging does not reduce TDE because FUGR-C is already very stable.
This script tests a different idea: do not average frames. Instead, attenuate only the FUGR residual
where the residual is temporally unreliable.

FUGR_t = Basic_t + R_t
TRG_t  = Basic_t + gamma_t * R_t

gamma_t is high when R_t agrees with flow-aligned neighboring residuals, and lower where the residual
appears temporally unstable. This aims to reduce residual flicker while preserving BasicVSR structure.
"""

import argparse, csv
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
    return float(np.mean([np.mean(np.abs((xs[i] - xs[i - 1]) - (ys[i] - ys[i - 1]))) for i in range(1, len(xs))]))


def motion(ys):
    if len(ys) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(ys[i] - ys[i - 1])) for i in range(1, len(ys))]))


def flow_cur_to_ref(cur, ref):
    return cv2.calcOpticalFlowFarneback(
        gray(cur), gray(ref), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )


def warp(ref, flow):
    h, w = flow.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    mx = (xx + flow[..., 0]).astype(np.float32)
    my = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(ref, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


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


def panel(path, imgs, names):
    imgs = [resize_h(x) for x in imgs]
    title_h = 32
    H = imgs[0].shape[0]
    W = sum(x.shape[1] for x in imgs)
    canvas = np.ones((H + title_h, W, 3), dtype=np.uint8) * 255
    x0 = 0
    for im, name in zip(imgs, names):
        w = im.shape[1]
        canvas[title_h:, x0:x0 + w] = np.clip(im * 255, 0, 255).round().astype(np.uint8)
        cv2.putText(canvas, name, (x0 + 5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
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
            gt_path = sd / f"{frame}_gt.png"
            if basic_path.exists() and gt_path.exists():
                items.append((frame, basic_path, fugr_path, gt_path))
        if items:
            data[sd.name] = items
    return data


def residual(basic, fugr):
    return [f - b for b, f in zip(basic, fugr)]


def residual_risk_gate(basic, fugr, i, tau, gamma_min, motion_tau, blur_sigma):
    res = residual(basic, fugr)
    cur_basic = basic[i]
    cur_res = res[i]

    risk_maps = []
    motion_maps = []
    for j in (i - 1, i + 1):
        if j < 0 or j >= len(res):
            continue
        fl = flow_cur_to_ref(cur_basic, basic[j])
        wr = warp(res[j], fl)
        diff = np.mean(np.abs(cur_res - wr), axis=2)
        mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
        risk_maps.append(diff)
        motion_maps.append(mag)

    if not risk_maps:
        gamma = np.ones(cur_basic.shape[:2], dtype=np.float32)
    else:
        risk = np.mean(risk_maps, axis=0)
        mag = np.mean(motion_maps, axis=0)
        consistency = np.exp(-risk / tau)
        motion_gate = np.exp(-mag / motion_tau)
        gamma = gamma_min + (1.0 - gamma_min) * consistency * motion_gate
        gamma = np.clip(cv2.GaussianBlur(gamma, (0, 0), blur_sigma), gamma_min, 1.0)

    out = np.clip(cur_basic + gamma[..., None] * cur_res, 0, 1)
    return out, gamma


def eval_seq(xs, ys):
    return dict(
        psnr=float(np.mean([psnr(x, y) for x, y in zip(xs, ys)])),
        ssim=float(np.mean([ssim_rgb(x, y) for x, y in zip(xs, ys)])),
        laplacian_sharpness=float(np.mean([sharpness(x) for x in xs])),
        tde=tde(xs, ys),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--taus", nargs="+", type=float, default=[0.002, 0.005, 0.010, 0.020, 0.040])
    ap.add_argument("--gamma_mins", nargs="+", type=float, default=[0.00, 0.30, 0.50, 0.70, 0.85, 0.95])
    ap.add_argument("--motion_taus", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--blur_sigmas", nargs="+", type=float, default=[0.0, 0.8, 1.5])
    ap.add_argument("--panel_limit", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out_dir)
    metrics_dir = out / "metrics"
    panels_dir = out / "panels"
    frames_dir = out / "best_frames"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    raw = collect(args.input_dir)
    print("Sequences:", sorted(raw.keys()), flush=True)

    data = {}
    for seq, items in raw.items():
        data[seq] = {
            "frames": [x[0] for x in items],
            "basic": [read_rgb(x[1]) for x in items],
            "fugr": [read_rgb(x[2]) for x in items],
            "gt": [read_rgb(x[3]) for x in items],
        }

    seq_rows = []
    frame_rows = []

    def add_method(name, outputs):
        for seq, xs in outputs.items():
            ys = data[seq]["gt"]
            met = eval_seq(xs, ys)
            seq_rows.append({"sequence": seq, "method": name, "num_frames": len(xs), "motion": motion(ys), **met})
            for fr, x, y in zip(data[seq]["frames"], xs, ys):
                frame_rows.append({"sequence": seq, "frame": fr, "method": name,
                                   "psnr": psnr(x, y), "ssim": ssim_rgb(x, y),
                                   "laplacian_sharpness": sharpness(x)})

    add_method("BasicVSR", {s: data[s]["basic"] for s in data})
    add_method("FUGR-C", {s: data[s]["fugr"] for s in data})

    total = len(args.taus) * len(args.gamma_mins) * len(args.motion_taus) * len(args.blur_sigmas)
    k = 0
    for tau in args.taus:
        for gmin in args.gamma_mins:
            for mtau in args.motion_taus:
                for blur in args.blur_sigmas:
                    k += 1
                    name = f"TRG-t{tau:.3f}-g{gmin:.2f}-m{mtau:.1f}-b{blur:.1f}"
                    print(f"[{k}/{total}] {name}", flush=True)
                    outs = {}
                    for seq in data:
                        basics = data[seq]["basic"]
                        fugrs = data[seq]["fugr"]
                        outs[seq] = [residual_risk_gate(basics, fugrs, i, tau, gmin, mtau, blur)[0]
                                     for i in range(len(fugrs))]
                    add_method(name, outs)

    with (metrics_dir / "directionA3_sequence_metrics.csv").open("w", newline="") as f:
        fields = ["sequence", "method", "num_frames", "motion", "psnr", "ssim", "laplacian_sharpness", "tde"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(seq_rows)

    with (metrics_dir / "directionA3_frame_metrics.csv").open("w", newline="") as f:
        fields = ["sequence", "frame", "method", "psnr", "ssim", "laplacian_sharpness"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)

    base = [r for r in seq_rows if r["method"] == "FUGR-C"]
    bpsnr = float(np.mean([r["psnr"] for r in base]))
    btde = float(np.mean([r["tde"] for r in base]))

    summary = []
    for method in sorted(set(r["method"] for r in seq_rows)):
        rows = [r for r in seq_rows if r["method"] == method]
        rec = {
            "method": method,
            "num_sequences": len(rows),
            "num_frames": int(sum(r["num_frames"] for r in rows)),
            "motion": float(np.mean([r["motion"] for r in rows])),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "laplacian_sharpness": float(np.mean([r["laplacian_sharpness"] for r in rows])),
            "tde": float(np.mean([r["tde"] for r in rows])),
        }
        rec["delta_psnr_vs_fugr"] = rec["psnr"] - bpsnr
        rec["delta_tde_vs_fugr"] = rec["tde"] - btde
        rec["tde_reduction_pct"] = 100 * (btde - rec["tde"]) / btde if btde > 0 else 0.0
        summary.append(rec)

    summary.sort(key=lambda r: (r["tde"], -r["psnr"]))

    with (metrics_dir / "directionA3_summary_metrics.csv").open("w", newline="") as f:
        fields = ["method", "num_sequences", "num_frames", "motion", "psnr", "ssim",
                  "laplacian_sharpness", "tde", "delta_psnr_vs_fugr",
                  "delta_tde_vs_fugr", "tde_reduction_pct"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    candidates = [r for r in summary if r["method"] != "FUGR-C" and r["delta_psnr_vs_fugr"] > -0.05]
    best = candidates[0] if candidates else summary[0]

    txt = metrics_dir / "directionA3_best_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A3: Temporal-Risk Residual Gate\n\n")
        f.write(f"FUGR-C baseline: PSNR={bpsnr:.6f}, TDE={btde:.8f}\n\n")
        f.write("Top 25 methods by TDE:\n")
        for r in summary[:25]:
            f.write(f"{r['method']},{r['num_frames']},{r['psnr']:.6f},{r['ssim']:.6f},"
                    f"{r['laplacian_sharpness']:.8f},{r['tde']:.8f},"
                    f"dPSNR={r['delta_psnr_vs_fugr']:.6f},"
                    f"TDEred={r['tde_reduction_pct']:.3f}%\n")
        f.write("\nSelected best under PSNR-loss constraint:\n")
        f.write(str(best) + "\n")

    print(txt.read_text(), flush=True)

    best_name = best["method"]

    def make_best(seq):
        basics = data[seq]["basic"]
        fugrs = data[seq]["fugr"]
        if best_name == "BasicVSR":
            return basics, [np.ones(basics[0].shape[:2], dtype=np.float32) for _ in basics]
        if best_name == "FUGR-C":
            return fugrs, [np.ones(basics[0].shape[:2], dtype=np.float32) for _ in basics]
        parts = best_name.split("-")
        tau = float(parts[1][1:])
        gmin = float(parts[2][1:])
        mtau = float(parts[3][1:])
        blur = float(parts[4][1:])
        outs, gates = [], []
        for i in range(len(fugrs)):
            o, gate = residual_risk_gate(basics, fugrs, i, tau, gmin, mtau, blur)
            outs.append(o)
            gates.append(gate)
        return outs, gates

    panel_count = 0
    for seq in data:
        outs, gates = make_best(seq)
        for fr, out_img, gate, basic, fugr, gt in zip(data[seq]["frames"], outs, gates,
                                                     data[seq]["basic"], data[seq]["fugr"], data[seq]["gt"]):
            save_rgb(frames_dir / seq / f"{fr}_directionA3.png", out_img)
            save_rgb(frames_dir / seq / f"{fr}_gamma.png", colorize(gate))
            if panel_count < args.panel_limit:
                panel(panels_dir / f"panel_{seq}_{fr}.png",
                      [basic, fugr, out_img, gt, err_map(fugr, gt), err_map(out_img, gt), colorize(gate)],
                      ["Basic", "FUGR-C", "A3-TRG", "GT", "FUGR Err", "A3 Err", "Gamma"])
                panel_count += 1

    print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
