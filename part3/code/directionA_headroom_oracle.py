#!/usr/bin/env python3
"""
Direction A4: Headroom / Oracle Blend Analysis.

Goal:
  Before designing more temporal post-processing, test whether there is any
  simple tradeoff headroom between BasicVSR and FUGR-C.

Family:
  output(lambda) = BasicVSR + lambda * (FUGR-C - BasicVSR)

Interpretation:
  lambda = 0: BasicVSR
  lambda = 1: FUGR-C
  lambda < 1: attenuate FUGR residual
  lambda > 1: amplify FUGR residual

Analyses:
  A4.1 Global lambda sweep on all clips.
  A4.2 Per-sequence oracle lambda, as an upper-bound headroom check.
  A4.3 Calibration/test split: select lambda on calibration sequences and evaluate on test sequences.
  A4.4 PSNR-TDE Pareto plot and lambda curves.

Input:
  input_dir/frames/<seq>/<frame>_basic.png
  input_dir/frames/<seq>/<frame>_fugr.png
  input_dir/frames/<seq>/<frame>_gt.png
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
    u8 = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(p), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def gray(img):
    return cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


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


def motion(ys):
    if len(ys) < 2:
        return 0.0
    vals = []
    for i in range(1, len(ys)):
        vals.append(float(np.mean(np.abs(ys[i] - ys[i - 1]))))
    return float(np.mean(vals))


def blend(basic, fugr, lam):
    return np.clip(basic + lam * (fugr - basic), 0, 1)


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


def eval_sequence(items, lam):
    outs, gts = [], []
    for it in items:
        out = blend(it["basic"], it["fugr"], lam)
        outs.append(out)
        gts.append(it["gt"])
    return {
        "num_frames": len(outs),
        "motion": motion(gts),
        "psnr": float(np.mean([psnr(o, g) for o, g in zip(outs, gts)])),
        "ssim": float(np.mean([ssim_rgb(o, g) for o, g in zip(outs, gts)])),
        "sharpness": float(np.mean([sharpness(o) for o in outs])),
        "tde": tde(outs, gts),
    }


def aggregate(seq_rows, split):
    return {
        "split": split,
        "lambda": seq_rows[0]["lambda"],
        "num_sequences": len(seq_rows),
        "num_frames": int(sum(r["num_frames"] for r in seq_rows)),
        "motion": float(np.mean([r["motion"] for r in seq_rows])),
        "psnr": float(np.mean([r["psnr"] for r in seq_rows])),
        "ssim": float(np.mean([r["ssim"] for r in seq_rows])),
        "sharpness": float(np.mean([r["sharpness"] for r in seq_rows])),
        "tde": float(np.mean([r["tde"] for r in seq_rows])),
    }


def eval_split(data, seqs, lambdas, split):
    seq_rows = []
    summary_rows = []
    for lam in lambdas:
        current = []
        for seq in seqs:
            rec = eval_sequence(data[seq], lam)
            rec.update({"split": split, "sequence": seq, "lambda": lam})
            current.append(rec)
            seq_rows.append(rec)
        summary_rows.append(aggregate(current, split))
    return seq_rows, summary_rows


def add_delta(summary_rows, baseline_lambda=1.0):
    by_split = {}
    for r in summary_rows:
        if abs(float(r["lambda"]) - baseline_lambda) < 1e-9:
            by_split[r["split"]] = r
    for r in summary_rows:
        base = by_split[r["split"]]
        r["delta_psnr_vs_fugr"] = r["psnr"] - base["psnr"]
        r["delta_ssim_vs_fugr"] = r["ssim"] - base["ssim"]
        r["delta_sharpness_vs_fugr"] = r["sharpness"] - base["sharpness"]
        r["delta_tde_vs_fugr"] = r["tde"] - base["tde"]
        r["tde_reduction_pct_vs_fugr"] = 100.0 * (base["tde"] - r["tde"]) / base["tde"] if base["tde"] > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_seqs", nargs="+", default=["003", "020"])
    ap.add_argument("--test_seqs", nargs="+", default=["028"])
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[-0.25, 0.00, 0.25, 0.50, 0.70, 0.85, 0.95, 1.00, 1.05, 1.10, 1.20, 1.50])
    ap.add_argument("--psnr_loss_limit", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    fig_dir = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = collect(args.input_dir)
    all_seqs = sorted(data.keys())
    print("Loaded sequences:", all_seqs, flush=True)

    missing = [s for s in args.calib_seqs + args.test_seqs if s not in data]
    if missing:
        raise FileNotFoundError(f"Missing sequences: {missing}")

    # A4.1 Global sweep over all available sequences.
    all_seq_rows, all_summary = eval_split(data, all_seqs, args.lambdas, "all")
    add_delta(all_summary)

    # A4.2 Per-sequence oracle.
    per_seq_oracle = []
    for seq in all_seqs:
        seq_rows, _ = eval_split(data, [seq], args.lambdas, f"seq_{seq}")
        fugr = next(r for r in seq_rows if abs(r["lambda"] - 1.0) < 1e-9)
        for r in seq_rows:
            r["delta_psnr_vs_fugr"] = r["psnr"] - fugr["psnr"]
            r["delta_tde_vs_fugr"] = r["tde"] - fugr["tde"]

        best_by_tde_under_psnr = sorted(
            [r for r in seq_rows if r["delta_psnr_vs_fugr"] >= -args.psnr_loss_limit],
            key=lambda r: (r["tde"], -r["psnr"])
        )[0]
        best_by_psnr = sorted(seq_rows, key=lambda r: (r["psnr"], -r["tde"]), reverse=True)[0]
        per_seq_oracle.append({
            "sequence": seq,
            "fugr_psnr": fugr["psnr"],
            "fugr_tde": fugr["tde"],
            "best_tde_lambda": best_by_tde_under_psnr["lambda"],
            "best_tde_psnr": best_by_tde_under_psnr["psnr"],
            "best_tde_tde": best_by_tde_under_psnr["tde"],
            "best_tde_delta_psnr_vs_fugr": best_by_tde_under_psnr["delta_psnr_vs_fugr"],
            "best_tde_delta_tde_vs_fugr": best_by_tde_under_psnr["delta_tde_vs_fugr"],
            "best_psnr_lambda": best_by_psnr["lambda"],
            "best_psnr": best_by_psnr["psnr"],
            "best_psnr_tde": best_by_psnr["tde"],
            "best_psnr_delta_vs_fugr": best_by_psnr["psnr"] - fugr["psnr"],
        })

    # A4.3 Calibration/test split.
    calib_seq_rows, calib_summary = eval_split(data, args.calib_seqs, args.lambdas, "calibration")
    test_seq_rows, test_summary = eval_split(data, args.test_seqs, args.lambdas, "test")
    split_summary = calib_summary + test_summary
    add_delta(split_summary)

    # Select lambda on calibration split by TDE under PSNR-loss constraint.
    calib_fugr = next(r for r in calib_summary if abs(r["lambda"] - 1.0) < 1e-9)
    candidates = [
        r for r in calib_summary
        if r["psnr"] - calib_fugr["psnr"] >= -args.psnr_loss_limit
    ]
    selected_by_tde = sorted(candidates, key=lambda r: (r["tde"], -r["psnr"]))[0]
    selected_by_psnr = sorted(calib_summary, key=lambda r: (r["psnr"], -r["tde"]), reverse=True)[0]

    selected_tde_test = next(r for r in test_summary if abs(r["lambda"] - selected_by_tde["lambda"]) < 1e-9)
    selected_psnr_test = next(r for r in test_summary if abs(r["lambda"] - selected_by_psnr["lambda"]) < 1e-9)
    test_fugr = next(r for r in test_summary if abs(r["lambda"] - 1.0) < 1e-9)

    split_selected_rows = [
        {"selection_rule": "calib_best_tde_under_psnr_constraint", **selected_by_tde},
        {"selection_rule": "same_lambda_on_test", **selected_tde_test},
        {"selection_rule": "calib_best_psnr", **selected_by_psnr},
        {"selection_rule": "same_lambda_on_test", **selected_psnr_test},
        {"selection_rule": "fugr_test_reference", **test_fugr},
    ]

    # Save CSVs.
    with (metrics_dir / "A4_global_lambda_sequence_metrics.csv").open("w", newline="") as f:
        fields = ["split", "sequence", "lambda", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_seq_rows)

    with (metrics_dir / "A4_global_lambda_summary.csv").open("w", newline="") as f:
        fields = ["split", "lambda", "num_sequences", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_sharpness_vs_fugr",
                  "delta_tde_vs_fugr", "tde_reduction_pct_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(all_summary, key=lambda r: (r["tde"], -r["psnr"])))

    with (metrics_dir / "A4_per_sequence_oracle.csv").open("w", newline="") as f:
        fields = ["sequence", "fugr_psnr", "fugr_tde",
                  "best_tde_lambda", "best_tde_psnr", "best_tde_tde",
                  "best_tde_delta_psnr_vs_fugr", "best_tde_delta_tde_vs_fugr",
                  "best_psnr_lambda", "best_psnr", "best_psnr_tde", "best_psnr_delta_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_seq_oracle)

    with (metrics_dir / "A4_calib_test_lambda_summary.csv").open("w", newline="") as f:
        fields = ["split", "lambda", "num_sequences", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_sharpness_vs_fugr",
                  "delta_tde_vs_fugr", "tde_reduction_pct_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(split_summary, key=lambda r: (r["split"], r["lambda"])))

    with (metrics_dir / "A4_calib_selected_test_result.csv").open("w", newline="") as f:
        fields = ["selection_rule", "split", "lambda", "num_sequences", "num_frames", "motion", "psnr", "ssim",
                  "sharpness", "tde", "delta_psnr_vs_fugr", "delta_ssim_vs_fugr",
                  "delta_sharpness_vs_fugr", "delta_tde_vs_fugr", "tde_reduction_pct_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(split_selected_rows)

    # Figures.
    all_sorted = sorted(all_summary, key=lambda r: r["lambda"])
    xs = [r["lambda"] for r in all_sorted]
    ps = [r["psnr"] for r in all_sorted]
    td = [r["tde"] for r in all_sorted]
    ss = [r["ssim"] for r in all_sorted]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ps, marker="o")
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("lambda")
    plt.ylabel("PSNR")
    plt.title("A4 Global lambda sweep: PSNR")
    plt.tight_layout()
    plt.savefig(fig_dir / "A4_lambda_psnr_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, td, marker="o")
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("lambda")
    plt.ylabel("TDE lower is better")
    plt.title("A4 Global lambda sweep: TDE")
    plt.tight_layout()
    plt.savefig(fig_dir / "A4_lambda_tde_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(td, ps)
    for x, y, lam in zip(td, ps, xs):
        plt.text(x, y, str(lam), fontsize=7)
    plt.xlabel("TDE lower is better")
    plt.ylabel("PSNR higher is better")
    plt.title("A4 PSNR-TDE tradeoff")
    plt.tight_layout()
    plt.savefig(fig_dir / "A4_pareto_psnr_tde.png", dpi=300)
    plt.close()

    # Summary txt.
    best_global_tde = sorted(
        [r for r in all_summary if r["delta_psnr_vs_fugr"] >= -args.psnr_loss_limit],
        key=lambda r: (r["tde"], -r["psnr"])
    )[0]
    best_global_psnr = sorted(all_summary, key=lambda r: (r["psnr"], -r["tde"]), reverse=True)[0]
    fugr_global = next(r for r in all_summary if abs(r["lambda"] - 1.0) < 1e-9)

    txt = metrics_dir / "A4_headroom_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A4: Headroom / Oracle Blend Analysis\n\n")
        f.write("Global FUGR-C reference:\n")
        f.write(str(fugr_global) + "\n\n")
        f.write("Best global TDE under PSNR-loss constraint:\n")
        f.write(str(best_global_tde) + "\n\n")
        f.write("Best global PSNR:\n")
        f.write(str(best_global_psnr) + "\n\n")
        f.write("Calibration-selected lambda by TDE under PSNR-loss constraint:\n")
        f.write(str(selected_by_tde) + "\n")
        f.write("Same lambda evaluated on test:\n")
        f.write(str(selected_tde_test) + "\n\n")
        f.write("Calibration-selected lambda by PSNR:\n")
        f.write(str(selected_by_psnr) + "\n")
        f.write("Same lambda evaluated on test:\n")
        f.write(str(selected_psnr_test) + "\n\n")
        f.write("Per-sequence oracle rows:\n")
        for r in per_seq_oracle:
            f.write(str(r) + "\n")

    print(txt.read_text())
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
