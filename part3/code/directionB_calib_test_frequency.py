#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import cv2
import numpy as np

def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def save_rgb(p, img):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
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

def ssim_rgb(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        x, y = a[..., ch].astype(np.float32), b[..., ch].astype(np.float32)
        mx, my = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x*x, (11, 11), 1.5) - mx*mx
        vy = cv2.GaussianBlur(y*y, (11, 11), 1.5) - my*my
        vxy = cv2.GaussianBlur(x*y, (11, 11), 1.5) - mx*my
        vals.append(float(np.mean(((2*mx*my+c1)*(2*vxy+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)+1e-12))))
    return float(np.mean(vals))

def sharpness(img):
    return float(np.var(cv2.Laplacian(gray(img), cv2.CV_32F)))

def tde(xs, ys):
    if len(xs) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs((xs[i]-xs[i-1])-(ys[i]-ys[i-1]))) for i in range(1, len(xs))]))

def collect(input_dir):
    root = Path(input_dir) / "frames"
    data = {}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        items = []
        for fugr_path in sorted(sd.glob("*_fugr.png")):
            frame = fugr_path.name.replace("_fugr.png", "")
            cn_path = sd / f"{frame}_controlnet_fugr.png"
            gt_path = sd / f"{frame}_gt.png"
            basic_path = sd / f"{frame}_basic.png"
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

def evaluate(data, seqs, method, make_img):
    seq_rows = []
    frame_rows = []
    for seq in seqs:
        outs, gts = [], []
        for it in data[seq]:
            out = make_img(it)
            gt = it["gt"]
            outs.append(out); gts.append(gt)
            frame_rows.append({
                "sequence": seq, "frame": it["frame"], "method": method,
                "psnr": psnr(out, gt), "ssim": ssim_rgb(out, gt), "sharpness": sharpness(out)
            })
        seq_rows.append({
            "sequence": seq, "method": method, "num_frames": len(outs),
            "psnr": float(np.mean([psnr(o, g) for o, g in zip(outs, gts)])),
            "ssim": float(np.mean([ssim_rgb(o, g) for o, g in zip(outs, gts)])),
            "sharpness": float(np.mean([sharpness(o) for o in outs])),
            "tde": tde(outs, gts),
        })
    return frame_rows, seq_rows

def aggregate(rows, split):
    return {
        "split": split,
        "method": rows[0]["method"],
        "num_sequences": len(rows),
        "num_frames": int(sum(r["num_frames"] for r in rows)),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "sharpness": float(np.mean([r["sharpness"] for r in rows])),
        "tde": float(np.mean([r["tde"] for r in rows])),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_seqs", nargs="+", default=["000", "003", "006", "010", "011"])
    ap.add_argument("--test_seqs", nargs="+", default=["018", "020", "026", "028", "029"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.8])
    ap.add_argument("--save_selected_frames", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    md = out / "metrics"
    fd = out / "selected_test_frames"
    md.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)

    data = collect(args.input_dir)
    missing = [s for s in args.calib_seqs + args.test_seqs if s not in data]
    if missing:
        raise FileNotFoundError(f"Missing sequences in input_dir: {missing}")

    all_summary, all_seq, all_frame = [], [], []

    for split, seqs in [("calibration", args.calib_seqs), ("test", args.test_seqs)]:
        for method, fn in [
            ("FUGR-C", lambda it: it["fugr"]),
            ("ControlNet-FUGR", lambda it: it["cn"]),
            ("BasicVSR", lambda it: it["basic"]),
        ]:
            fr, sr = evaluate(data, seqs, method, fn)
            all_frame.extend([{"split": split, **r} for r in fr])
            all_seq.extend([{"split": split, **r} for r in sr])
            all_summary.append(aggregate(sr, split))

    config_rows = []
    for beta in args.betas:
        for sigma in args.sigmas:
            method = f"RGB-HF-b{beta:.2f}-s{sigma:.1f}"
            make = lambda it, b=beta, s=sigma: rgb_hf(it["fugr"], it["cn"], b, s)
            fr, sr = evaluate(data, args.calib_seqs, method, make)
            rec = aggregate(sr, "calibration")
            rec["beta"] = beta; rec["sigma"] = sigma
            config_rows.append(rec)
            all_frame.extend([{"split": "calibration", **r} for r in fr])
            all_seq.extend([{"split": "calibration", **r} for r in sr])
            all_summary.append(rec)

    configs_sorted = sorted(config_rows, key=lambda r: (r["psnr"], r["ssim"], -r["tde"]), reverse=True)
    best = configs_sorted[0]
    beta, sigma = best["beta"], best["sigma"]
    best_method = f"RGB-HF-b{beta:.2f}-s{sigma:.1f}"
    make_best = lambda it: rgb_hf(it["fugr"], it["cn"], beta, sigma)

    fr, sr = evaluate(data, args.test_seqs, best_method, make_best)
    test_best = aggregate(sr, "test")
    test_best["beta"] = beta; test_best["sigma"] = sigma
    all_frame.extend([{"split": "test", **r} for r in fr])
    all_seq.extend([{"split": "test", **r} for r in sr])
    all_summary.append(test_best)

    if args.save_selected_frames:
        for seq in args.test_seqs:
            for it in data[seq]:
                save_rgb(fd / seq / f"{it['frame']}_selected_freqfusion.png", make_best(it))

    fugr = {r["split"]: r for r in all_summary if r["method"] == "FUGR-C"}
    for r in all_summary:
        base = fugr[r["split"]]
        r["delta_psnr_vs_fugr"] = r["psnr"] - base["psnr"]
        r["delta_ssim_vs_fugr"] = r["ssim"] - base["ssim"]
        r["delta_tde_vs_fugr"] = r["tde"] - base["tde"]

    with (md / "B15_calib_config_ranking.csv").open("w", newline="") as f:
        fields = ["split","method","num_sequences","num_frames","psnr","ssim","sharpness","tde",
                  "delta_psnr_vs_fugr","delta_ssim_vs_fugr","delta_tde_vs_fugr",
                  "beta","sigma"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(configs_sorted)

    with (md / "B15_all_summary.csv").open("w", newline="") as f:
        fields = ["split","method","num_sequences","num_frames","psnr","ssim","sharpness","tde",
                  "delta_psnr_vs_fugr","delta_ssim_vs_fugr","delta_tde_vs_fugr","beta","sigma"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_summary)

    with (md / "B15_sequence_metrics.csv").open("w", newline="") as f:
        fields = ["split","sequence","method","num_frames","psnr","ssim","sharpness","tde"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_seq)

    with (md / "B15_frame_metrics.csv").open("w", newline="") as f:
        fields = ["split","sequence","frame","method","psnr","ssim","sharpness"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_frame)

    txt = md / "B15_calib_test_summary.txt"
    with txt.open("w") as f:
        f.write("Direction B15: Calibration/Test Frequency Fusion\n\n")
        f.write(f"Calibration sequences: {args.calib_seqs}\n")
        f.write(f"Test sequences: {args.test_seqs}\n\n")
        f.write("Selected config on calibration split:\n")
        f.write(str(best) + "\n\n")
        f.write("Selected config evaluated on held-out test split:\n")
        f.write(str(test_best) + "\n\n")
        f.write("Key rows:\n")
        for r in all_summary:
            if r["method"] in ["FUGR-C", "ControlNet-FUGR", "BasicVSR", best_method]:
                f.write(str(r) + "\n")

    print(txt.read_text())

if __name__ == "__main__":
    main()
