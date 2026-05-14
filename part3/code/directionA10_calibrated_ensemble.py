#!/usr/bin/env python3
"""
Direction A10: calibrated residual ensemble exploration.

Motivation:
  A9 learned a no-reference selector, but selection is a hard decision and may be too noisy
  when the true headroom is extremely small. A10 tests a stronger supervised-calibration idea:
  learn a global residual correction from multiple temporal candidates on calibration frames,
  then apply it to held-out test sequences.

Core model:
  y = GT - FUGR
  X_j = Candidate_j - FUGR
  Learn ridge weights w so that sum_j w_j X_j approximates y on calibration frames.
  Test output = FUGR + scale * sum_j w_j (Candidate_j - FUGR)

This is still post-hoc and deployable after calibration:
  - GT is used only for calibration split.
  - Test uses only BasicVSR/FUGR candidate outputs.

Outputs:
  A10_calibrated_ensemble_summary.txt
  A10_policy_summary.csv
  A10_sequence_details.csv
  A10_weights.csv
  A10_policy_delta_psnr.png
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
    return float(np.mean([np.mean(np.abs((xs[i] - xs[i - 1]) - (ys[i] - ys[i - 1])))
                          for i in range(1, len(xs))]))


def collect(input_dir):
    root = Path(input_dir) / "frames"
    data = {}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        items = []
        for fp in sorted(sd.glob("*_fugr.png")):
            frame = fp.name.replace("_fugr.png", "")
            bp = sd / f"{frame}_basic.png"
            gp = sd / f"{frame}_gt.png"
            if bp.exists() and gp.exists():
                items.append({
                    "frame": frame,
                    "basic": read_rgb(bp),
                    "fugr": read_rgb(fp),
                    "gt": read_rgb(gp),
                })
        if items:
            data[sd.name] = items
    return data


def residuals(basic, fugr):
    return [f - b for b, f in zip(basic, fugr)]


def make_lambda(basic, fugr, lam):
    return [np.clip(b + lam * (f - b), 0, 1) for b, f in zip(basic, fugr)]


def make_atten(basic, fugr, tau=0.020, gmin=0.95, blur=0.8):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        neigh = []
        if i > 0:
            neigh.append(rs[i - 1])
        if i + 1 < len(rs):
            neigh.append(rs[i + 1])
        if neigh:
            avg = np.mean(np.stack(neigh, axis=0), axis=0)
            risk = np.mean(np.abs(r - avg), axis=2)
            gamma = gmin + (1 - gmin) * np.exp(-risk / tau)
            if blur > 0:
                gamma = cv2.GaussianBlur(gamma, (0, 0), blur)
            gamma = np.clip(gamma, gmin, 1.0)
        else:
            gamma = np.ones(r.shape[:2], dtype=np.float32)
        outs.append(np.clip(basic[i] + gamma[..., None] * r, 0, 1))
    return outs


def make_median(basic, fugr, beta=0.25):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        neigh = [r]
        if i > 0:
            neigh.append(rs[i - 1])
        if i + 1 < len(rs):
            neigh.append(rs[i + 1])
        med = np.median(np.stack(neigh, axis=0), axis=0)
        outs.append(np.clip(basic[i] + (1 - beta) * r + beta * med, 0, 1))
    return outs


def make_clip(basic, fugr, scale=1.2):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        neigh = [np.abs(r)]
        if i > 0:
            neigh.append(np.abs(rs[i - 1]))
        if i + 1 < len(rs):
            neigh.append(np.abs(rs[i + 1]))
        lim = scale * np.median(np.stack(neigh, axis=0), axis=0) + 1e-4
        outs.append(np.clip(basic[i] + np.clip(r, -lim, lim), 0, 1))
    return outs


def build_candidates(blur):
    # FUGR-C is the reference. The remaining candidates form the residual basis.
    return [
        ("Lambda-0.85", lambda b, f: make_lambda(b, f, 0.85)),
        ("Lambda-0.95", lambda b, f: make_lambda(b, f, 0.95)),
        ("Lambda-1.05", lambda b, f: make_lambda(b, f, 1.05)),
        ("Lambda-1.20", lambda b, f: make_lambda(b, f, 1.20)),
        ("ResidualAtten-t0.020-g0.95", lambda b, f: make_atten(b, f, 0.020, 0.95, blur)),
        ("ResidualAtten-t0.010-g0.95", lambda b, f: make_atten(b, f, 0.010, 0.95, blur)),
        ("ResidualMedian-b0.25", lambda b, f: make_median(b, f, 0.25)),
        ("ResidualClip-c1.20", lambda b, f: make_clip(b, f, 1.20)),
    ]


def eval_sequence(xs, gt):
    return {
        "num_frames": len(xs),
        "psnr": float(np.mean([psnr(x, y) for x, y in zip(xs, gt)])),
        "ssim": float(np.mean([ssim_rgb(x, y) for x, y in zip(xs, gt)])),
        "sharpness": float(np.mean([sharpness(x) for x in xs])),
        "tde": tde(xs, gt),
    }


def ridge_fit(X, y, alpha):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def make_outputs_for_seq(items, candidates):
    basic = [it["basic"] for it in items]
    fugr = [it["fugr"] for it in items]
    gt = [it["gt"] for it in items]
    cand = {name: fn(basic, fugr) for name, fn in candidates}
    return basic, fugr, gt, cand


def sample_training_rows(data, calib_seqs, candidates, pixels_per_frame, seed):
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []

    for seq in calib_seqs:
        basic, fugr, gt, cand = make_outputs_for_seq(data[seq], candidates)
        names = list(cand.keys())
        for i in range(len(gt)):
            h, w = gt[i].shape[:2]
            n = min(pixels_per_frame, h * w)
            flat_idx = rng.choice(h * w, size=n, replace=False)
            yy = flat_idx // w
            xx = flat_idx % w

            # For each sampled pixel and color channel, feature = candidate residuals vs FUGR.
            for ch in range(3):
                cols = []
                for name in names:
                    cols.append((cand[name][i][yy, xx, ch] - fugr[i][yy, xx, ch]).reshape(-1, 1))
                X = np.concatenate(cols, axis=1)
                y = (gt[i][yy, xx, ch] - fugr[i][yy, xx, ch]).reshape(-1)
                X_list.append(X)
                y_list.append(y)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def residual_correction_outputs(fugr, cand, names, weights, scale):
    outs = []
    for i in range(len(fugr)):
        corr = np.zeros_like(fugr[i])
        for w, name in zip(weights, names):
            corr += float(w) * (cand[name][i] - fugr[i])
        outs.append(np.clip(fugr[i] + scale * corr, 0, 1))
    return outs


def add_policy(policy_rows, detail_rows, policy, seqs, data, candidates, output_fn):
    seq_eval = []
    for seq in seqs:
        basic, fugr, gt, cand = make_outputs_for_seq(data[seq], candidates)
        outs = output_fn(seq, basic, fugr, gt, cand)
        rec = eval_sequence(outs, gt)
        rec.update({"policy": policy, "sequence": seq})
        seq_eval.append(rec)

        fr = eval_sequence(fugr, gt)
        detail_rows.append({
            "policy": policy,
            "sequence": seq,
            "psnr": rec["psnr"],
            "ssim": rec["ssim"],
            "sharpness": rec["sharpness"],
            "tde": rec["tde"],
            "fugr_psnr": fr["psnr"],
            "fugr_tde": fr["tde"],
            "delta_psnr_vs_fugr": rec["psnr"] - fr["psnr"],
            "delta_tde_vs_fugr": rec["tde"] - fr["tde"],
        })

    agg = {
        "policy": policy,
        "num_sequences": len(seq_eval),
        "num_frames": int(sum(r["num_frames"] for r in seq_eval)),
        "psnr": float(np.mean([r["psnr"] for r in seq_eval])),
        "ssim": float(np.mean([r["ssim"] for r in seq_eval])),
        "sharpness": float(np.mean([r["sharpness"] for r in seq_eval])),
        "tde": float(np.mean([r["tde"] for r in seq_eval])),
    }

    fugr_eval = []
    for seq in seqs:
        _, fugr, gt, _ = make_outputs_for_seq(data[seq], candidates)
        fugr_eval.append(eval_sequence(fugr, gt))
    fugr_agg = {
        "psnr": float(np.mean([r["psnr"] for r in fugr_eval])),
        "tde": float(np.mean([r["tde"] for r in fugr_eval])),
    }
    agg["delta_psnr_vs_fugr"] = agg["psnr"] - fugr_agg["psnr"]
    agg["delta_tde_vs_fugr"] = agg["tde"] - fugr_agg["tde"]
    policy_rows.append(agg)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_seqs", nargs="+", default=["000", "003", "006", "010", "011"])
    ap.add_argument("--test_seqs", nargs="+", default=["018", "020", "026", "028", "029"])
    ap.add_argument("--blur_sigma", type=float, default=0.8)
    ap.add_argument("--ridge_alpha", type=float, default=1e-3)
    ap.add_argument("--pixels_per_frame", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.00])
    args = ap.parse_args()

    out = Path(args.out_dir)
    md = out / "metrics"
    fd = out / "figures"
    md.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)

    data = collect(args.input_dir)
    candidates = build_candidates(args.blur_sigma)
    names = [n for n, _ in candidates]

    print("Loaded sequences:", sorted(data.keys()), flush=True)
    print("Candidate basis:", names, flush=True)

    # Train ridge residual ensemble on calibration.
    print("Sampling calibration pixels...", flush=True)
    X, y = sample_training_rows(data, args.calib_seqs, candidates, args.pixels_per_frame, args.seed)
    print("Training rows:", X.shape, "target:", y.shape, flush=True)
    weights = ridge_fit(X, y, args.ridge_alpha)

    # Evaluate policies.
    policy_rows = []
    detail_rows = []

    add_policy(policy_rows, detail_rows, "FUGR-C", args.test_seqs, data, candidates,
               lambda seq, basic, fugr, gt, cand: fugr)

    # Fixed learned residual correction scales.
    for scale in args.scales:
        add_policy(
            policy_rows, detail_rows,
            f"RidgeResidualEnsemble-scale{scale:.2f}",
            args.test_seqs, data, candidates,
            lambda seq, basic, fugr, gt, cand, scale=scale:
                residual_correction_outputs(fugr, cand, names, weights, scale)
        )

    # Select best scale on calibration by PSNR, then evaluate on test.
    calib_scale_rows = []
    dummy_details = []
    for scale in args.scales:
        row = add_policy(
            calib_scale_rows, dummy_details,
            f"calib-scale{scale:.2f}",
            args.calib_seqs, data, candidates,
            lambda seq, basic, fugr, gt, cand, scale=scale:
                residual_correction_outputs(fugr, cand, names, weights, scale)
        )
    best_calib_psnr = max(calib_scale_rows, key=lambda r: (r["psnr"], -r["tde"]))
    best_scale = float(best_calib_psnr["policy"].replace("calib-scale", ""))

    add_policy(
        policy_rows, detail_rows,
        f"CalibSelectedScale-{best_scale:.2f}",
        args.test_seqs, data, candidates,
        lambda seq, basic, fugr, gt, cand, scale=best_scale:
            residual_correction_outputs(fugr, cand, names, weights, scale)
    )

    # Oracle best scale on test for upper bound of this learned correction direction.
    test_scale_rows = []
    dummy_details = []
    for scale in args.scales:
        add_policy(
            test_scale_rows, dummy_details,
            f"test-scale{scale:.2f}",
            args.test_seqs, data, candidates,
            lambda seq, basic, fugr, gt, cand, scale=scale:
                residual_correction_outputs(fugr, cand, names, weights, scale)
        )
    best_test_scale_row = max(test_scale_rows, key=lambda r: (r["psnr"], -r["tde"]))

    # Save CSVs.
    def write_csv(path, rows, fields):
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    weight_rows = [{"candidate": n, "weight": float(w)} for n, w in zip(names, weights)]
    write_csv(md / "A10_weights.csv", weight_rows, ["candidate", "weight"])

    fields = ["policy", "num_sequences", "num_frames", "psnr", "ssim", "sharpness", "tde",
              "delta_psnr_vs_fugr", "delta_tde_vs_fugr"]
    write_csv(md / "A10_policy_summary.csv", policy_rows, fields)
    write_csv(md / "A10_calib_scale_summary.csv", calib_scale_rows, fields)
    write_csv(md / "A10_test_scale_oracle_summary.csv", test_scale_rows, fields)

    write_csv(md / "A10_sequence_details.csv", detail_rows,
              ["policy", "sequence", "psnr", "ssim", "sharpness", "tde",
               "fugr_psnr", "fugr_tde", "delta_psnr_vs_fugr", "delta_tde_vs_fugr"])

    # Figure.
    order = sorted(policy_rows, key=lambda r: r["delta_psnr_vs_fugr"], reverse=True)
    plt.figure(figsize=(9, 4))
    plt.bar([r["policy"] for r in order], [r["delta_psnr_vs_fugr"] for r in order])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.ylabel("PSNR delta vs FUGR-C")
    plt.title("A10 calibrated residual ensemble on held-out test")
    plt.tight_layout()
    plt.savefig(fd / "A10_policy_delta_psnr.png", dpi=300)
    plt.close()

    best_deployable = max(policy_rows, key=lambda r: (r["psnr"], -r["tde"]))
    fugr_ref = next(r for r in policy_rows if r["policy"] == "FUGR-C")

    txt = md / "A10_calibrated_ensemble_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A10: Calibrated Residual Ensemble Exploration\n\n")
        f.write("FUGR-C test reference:\n")
        f.write(str(fugr_ref) + "\n\n")
        f.write("Learned residual basis weights:\n")
        for r in weight_rows:
            f.write(str(r) + "\n")
        f.write("\nBest calibration-selected scale:\n")
        f.write(str(best_calib_psnr) + "\n")
        f.write(f"selected scale: {best_scale:.2f}\n\n")
        f.write("Best deployable policy on test:\n")
        f.write(str(best_deployable) + "\n\n")
        f.write("Best test-oracle scale within learned correction family:\n")
        f.write(str(best_test_scale_row) + "\n\n")
        f.write("Interpretation:\n")
        f.write(
            "A10 tests a stronger supervised-calibration variant: instead of selecting one candidate, "
            "it learns a global residual correction from multiple temporal candidates on calibration frames. "
            "If the calibration-selected or even test-oracle correction scale fails to materially beat FUGR-C, "
            "then the remaining post-hoc temporal/refinement headroom is too small even for a calibrated linear ensemble.\n"
        )

    print(txt.read_text())
    print("Saved:", out)


if __name__ == "__main__":
    main()
