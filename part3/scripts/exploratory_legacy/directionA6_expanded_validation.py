#!/usr/bin/env python3
"""
Direction A6: Expanded validation for temporal refinement candidates.

Purpose
-------
A1-A5 were mostly performed on three short continuous clips. A6 checks whether the
same conclusion holds on a wider 10-sequence / 40-frame sampled benchmark that is
already available from DirectionB/expanded_strong_st015.

This script does NOT rerun BasicVSR/FUGR. It reuses saved frames:
  input_dir/frames/<seq>/<frame>_basic.png
  input_dir/frames/<seq>/<frame>_fugr.png
  input_dir/frames/<seq>/<frame>_gt.png

It evaluates:
  - BasicVSR
  - FUGR-C
  - residual lambda scaling
  - residual median stabilization
  - residual attenuation
  - residual clipping

It reports:
  A6.1 all-sequence summary
  A6.2 calibration/test split selection
  A6.3 leave-one-sequence-out selection robustness
  A6.4 PSNR-TDE tradeoff figures
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
    r = residuals(basic, fugr)
    outs = []
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


def build_methods(args):
    methods = [
        ("BasicVSR", lambda b, f: b),
        ("FUGR-C", lambda b, f: f),
    ]

    for lam in args.lambdas:
        methods.append((f"Lambda-{lam:.2f}", lambda b, f, lam=lam: make_lambda(b, f, lam)))

    for beta in args.median_betas:
        methods.append((f"ResidualMedian-b{beta:.2f}",
                        lambda b, f, beta=beta: make_residual_median(b, f, beta)))

    for tau in args.taus:
        for gamma_min in args.gamma_mins:
            methods.append((f"ResidualAtten-t{tau:.3f}-g{gamma_min:.2f}",
                            lambda b, f, tau=tau, gamma_min=gamma_min:
                            make_residual_attenuation(b, f, tau, gamma_min, args.blur_sigma)))

    for clip_scale in args.clip_scales:
        methods.append((f"ResidualClip-c{clip_scale:.2f}",
                        lambda b, f, clip_scale=clip_scale:
                        make_residual_clip(b, f, clip_scale)))

    # De-duplicate if Lambda-1.00 duplicates FUGR-C; keeping it is okay but can clutter.
    return methods


def eval_outputs(xs, ys):
    return {
        "num_frames": len(xs),
        "motion": motion(ys),
        "psnr": float(np.mean([psnr(x, y) for x, y in zip(xs, ys)])),
        "ssim": float(np.mean([ssim_rgb(x, y) for x, y in zip(xs, ys)])),
        "sharpness": float(np.mean([sharpness(x) for x in xs])),
        "tde": tde(xs, ys),
    }


def eval_method_on_seq(data, seq, builder):
    items = data[seq]
    basic = [it["basic"] for it in items]
    fugr = [it["fugr"] for it in items]
    gt = [it["gt"] for it in items]
    outs = builder(basic, fugr)
    return eval_outputs(outs, gt)


def eval_all(data, seqs, methods):
    rows = []
    for method, builder in methods:
        for seq in seqs:
            rec = eval_method_on_seq(data, seq, builder)
            rec.update({"sequence": seq, "method": method})
            rows.append(rec)
    return rows


def aggregate(seq_rows, split):
    methods = sorted(set(r["method"] for r in seq_rows))
    out = []
    for method in methods:
        rows = [r for r in seq_rows if r["method"] == method]
        out.append({
            "split": split,
            "method": method,
            "num_sequences": len(rows),
            "num_frames": int(sum(r["num_frames"] for r in rows)),
            "motion": float(np.mean([r["motion"] for r in rows])),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "sharpness": float(np.mean([r["sharpness"] for r in rows])),
            "tde": float(np.mean([r["tde"] for r in rows])),
        })
    return out


def add_deltas(summary_rows):
    fugr_by_split = {
        r["split"]: r for r in summary_rows
        if r["method"] == "FUGR-C"
    }
    for r in summary_rows:
        base = fugr_by_split[r["split"]]
        r["delta_psnr_vs_fugr"] = r["psnr"] - base["psnr"]
        r["delta_ssim_vs_fugr"] = r["ssim"] - base["ssim"]
        r["delta_sharpness_vs_fugr"] = r["sharpness"] - base["sharpness"]
        r["delta_tde_vs_fugr"] = r["tde"] - base["tde"]
        r["tde_reduction_pct_vs_fugr"] = 100 * (base["tde"] - r["tde"]) / base["tde"] if base["tde"] > 0 else 0.0


def select_best(summary_rows, split, psnr_loss_limit, include_fugr=True):
    rows = [r for r in summary_rows if r["split"] == split]
    fugr = next(r for r in rows if r["method"] == "FUGR-C")
    cand = [
        r for r in rows
        if r["psnr"] - fugr["psnr"] >= -psnr_loss_limit
        and (include_fugr or r["method"] != "FUGR-C")
    ]
    return sorted(cand, key=lambda r: (r["tde"], -r["psnr"]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_seqs", nargs="+", default=["000", "003", "006", "010", "011"])
    ap.add_argument("--test_seqs", nargs="+", default=["018", "020", "026", "028", "029"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.70, 0.85, 0.95, 1.00, 1.05, 1.10, 1.20])
    ap.add_argument("--median_betas", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.005, 0.010, 0.020])
    ap.add_argument("--gamma_mins", nargs="+", type=float, default=[0.85, 0.95])
    ap.add_argument("--clip_scales", nargs="+", type=float, default=[0.80, 1.00, 1.20])
    ap.add_argument("--blur_sigma", type=float, default=0.8)
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

    methods = build_methods(args)
    print("Number of methods:", len(methods), flush=True)

    # A6.1 all-sequence evaluation
    print("A6.1 evaluating all sequences", flush=True)
    all_seq_rows = eval_all(data, all_seqs, methods)
    all_summary = aggregate(all_seq_rows, "all")

    # A6.2 calibration/test
    print("A6.2 calibration/test evaluation", flush=True)
    calib_seq_rows = eval_all(data, args.calib_seqs, methods)
    test_seq_rows = eval_all(data, args.test_seqs, methods)
    split_summary = aggregate(calib_seq_rows, "calibration") + aggregate(test_seq_rows, "test")

    all_summary_plus_split = all_summary + split_summary
    add_deltas(all_summary_plus_split)

    # Select on calibration.
    selected_calib_all = select_best(all_summary_plus_split, "calibration", args.psnr_loss_limit, include_fugr=True)
    selected_calib_nonfugr = select_best(all_summary_plus_split, "calibration", args.psnr_loss_limit, include_fugr=False)

    test_rows = [r for r in all_summary_plus_split if r["split"] == "test"]
    selected_test_all = next(r for r in test_rows if r["method"] == selected_calib_all["method"])
    selected_test_nonfugr = next(r for r in test_rows if r["method"] == selected_calib_nonfugr["method"])
    test_fugr = next(r for r in test_rows if r["method"] == "FUGR-C")

    selected_rows = [
        {"selection_rule": "best_including_fugr_on_calibration", **selected_calib_all},
        {"selection_rule": "same_method_on_test", **selected_test_all},
        {"selection_rule": "best_nonfugr_on_calibration", **selected_calib_nonfugr},
        {"selection_rule": "same_method_on_test", **selected_test_nonfugr},
        {"selection_rule": "fugr_test_reference", **test_fugr},
    ]

    # A6.3 LOSO
    print("A6.3 leave-one-sequence-out validation", flush=True)
    loso_rows = []
    for held in all_seqs:
        train_seqs = [s for s in all_seqs if s != held]
        train_seq_rows = eval_all(data, train_seqs, methods)
        held_seq_rows = eval_all(data, [held], methods)
        train_summary = aggregate(train_seq_rows, "train")
        held_summary = aggregate(held_seq_rows, "heldout")
        add_deltas(train_summary + held_summary)

        best_train_all = select_best(train_summary, "train", args.psnr_loss_limit, include_fugr=True)
        best_train_nonfugr = select_best(train_summary, "train", args.psnr_loss_limit, include_fugr=False)
        held_fugr = next(r for r in held_summary if r["method"] == "FUGR-C")
        held_all = next(r for r in held_summary if r["method"] == best_train_all["method"])
        held_nonfugr = next(r for r in held_summary if r["method"] == best_train_nonfugr["method"])

        for label, train_best, held_best in [
            ("including_fugr", best_train_all, held_all),
            ("nonfugr_only", best_train_nonfugr, held_nonfugr),
        ]:
            loso_rows.append({
                "selection_mode": label,
                "heldout_sequence": held,
                "selected_method": train_best["method"],
                "train_psnr": train_best["psnr"],
                "train_tde": train_best["tde"],
                "heldout_psnr": held_best["psnr"],
                "heldout_ssim": held_best["ssim"],
                "heldout_tde": held_best["tde"],
                "fugr_psnr": held_fugr["psnr"],
                "fugr_ssim": held_fugr["ssim"],
                "fugr_tde": held_fugr["tde"],
                "delta_psnr_vs_fugr": held_best["psnr"] - held_fugr["psnr"],
                "delta_ssim_vs_fugr": held_best["ssim"] - held_fugr["ssim"],
                "delta_tde_vs_fugr": held_best["tde"] - held_fugr["tde"],
            })

    # Save CSVs.
    with (metrics_dir / "A6_sequence_metrics.csv").open("w", newline="") as f:
        fields = ["sequence", "method", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_seq_rows)

    with (metrics_dir / "A6_all_summary.csv").open("w", newline="") as f:
        fields = ["split", "method", "num_sequences", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr",
                  "delta_sharpness_vs_fugr", "delta_tde_vs_fugr",
                  "tde_reduction_pct_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(all_summary, key=lambda r: (r["tde"], -r["psnr"])))

    with (metrics_dir / "A6_calib_test_summary.csv").open("w", newline="") as f:
        fields = ["split", "method", "num_sequences", "num_frames", "motion", "psnr", "ssim", "sharpness", "tde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr",
                  "delta_sharpness_vs_fugr", "delta_tde_vs_fugr",
                  "tde_reduction_pct_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(split_summary, key=lambda r: (r["split"], r["tde"], -r["psnr"])))

    with (metrics_dir / "A6_selected_calib_test.csv").open("w", newline="") as f:
        fields = ["selection_rule", "split", "method", "num_sequences", "num_frames", "motion", "psnr", "ssim",
                  "sharpness", "tde", "delta_psnr_vs_fugr", "delta_ssim_vs_fugr",
                  "delta_sharpness_vs_fugr", "delta_tde_vs_fugr", "tde_reduction_pct_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected_rows)

    with (metrics_dir / "A6_loso_generalization.csv").open("w", newline="") as f:
        fields = ["selection_mode", "heldout_sequence", "selected_method", "train_psnr", "train_tde",
                  "heldout_psnr", "heldout_ssim", "heldout_tde",
                  "fugr_psnr", "fugr_ssim", "fugr_tde",
                  "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_tde_vs_fugr"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(loso_rows)

    # Figures.
    all_summary_sorted = sorted(all_summary, key=lambda r: (r["tde"], -r["psnr"]))
    xs = [r["tde"] for r in all_summary_sorted]
    ys = [r["psnr"] for r in all_summary_sorted]
    labels = [r["method"] for r in all_summary_sorted]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=26)
    fugr = next(r for r in all_summary_sorted if r["method"] == "FUGR-C")
    plt.scatter([fugr["tde"]], [fugr["psnr"]], s=100, marker="*")
    for x, y, lab in zip(xs[:10], ys[:10], labels[:10]):
        plt.text(x, y, lab.replace("Residual", "R"), fontsize=6)
    plt.xlabel("TDE lower is better")
    plt.ylabel("PSNR higher is better")
    plt.title("A6 expanded validation: PSNR-TDE tradeoff")
    plt.tight_layout()
    plt.savefig(fig_dir / "A6_pareto_psnr_tde.png", dpi=300)
    plt.close()

    top = all_summary_sorted[:12]
    plt.figure(figsize=(9, 4))
    plt.bar([r["method"] for r in top], [r["tde"] for r in top])
    plt.xticks(rotation=75, ha="right", fontsize=7)
    plt.ylabel("TDE lower is better")
    plt.title("A6 top methods by TDE")
    plt.tight_layout()
    plt.savefig(fig_dir / "A6_top_tde_bar.png", dpi=300)
    plt.close()

    loso_non = [r for r in loso_rows if r["selection_mode"] == "nonfugr_only"]
    loso_inc = [r for r in loso_rows if r["selection_mode"] == "including_fugr"]

    def mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    txt = metrics_dir / "A6_expanded_summary.txt"
    best_all = all_summary_sorted[0]
    best_non = sorted(
        [r for r in all_summary if r["method"] != "FUGR-C" and r["delta_psnr_vs_fugr"] >= -args.psnr_loss_limit],
        key=lambda r: (r["tde"], -r["psnr"])
    )[0]

    with txt.open("w") as f:
        f.write("Direction A6: Expanded Temporal Candidate Validation\n\n")
        f.write("All-sequence FUGR-C reference:\n")
        f.write(str(fugr) + "\n\n")
        f.write("Best all-sequence method by TDE:\n")
        f.write(str(best_all) + "\n\n")
        f.write("Best non-FUGR method by TDE under PSNR-loss constraint:\n")
        f.write(str(best_non) + "\n\n")
        f.write("Calibration/test selected rows:\n")
        for r in selected_rows:
            f.write(str(r) + "\n")
        f.write("\nLOSO including-FUGR selection:\n")
        f.write(f"mean delta PSNR: {mean([r['delta_psnr_vs_fugr'] for r in loso_inc]):.8f}\n")
        f.write(f"mean delta TDE: {mean([r['delta_tde_vs_fugr'] for r in loso_inc]):.8f}\n")
        f.write(f"positive PSNR sequences: {sum(r['delta_psnr_vs_fugr'] > 0 for r in loso_inc)}/{len(loso_inc)}\n")
        f.write(f"negative TDE sequences: {sum(r['delta_tde_vs_fugr'] < 0 for r in loso_inc)}/{len(loso_inc)}\n")
        f.write("\nLOSO non-FUGR-only selection:\n")
        f.write(f"mean delta PSNR: {mean([r['delta_psnr_vs_fugr'] for r in loso_non]):.8f}\n")
        f.write(f"mean delta TDE: {mean([r['delta_tde_vs_fugr'] for r in loso_non]):.8f}\n")
        f.write(f"positive PSNR sequences: {sum(r['delta_psnr_vs_fugr'] > 0 for r in loso_non)}/{len(loso_non)}\n")
        f.write(f"negative TDE sequences: {sum(r['delta_tde_vs_fugr'] < 0 for r in loso_non)}/{len(loso_non)}\n")

    print(txt.read_text())
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
