#!/usr/bin/env python3
"""
Direction A9: learned no-reference frame-level selector.

A8 tested hand-designed sequence-level policies. A9 asks whether a small learned
selector, trained only on calibration frames, can exploit the tiny remaining
post-hoc temporal-refinement headroom.

It trains one ridge-regression model per non-FUGR candidate to predict frame-level
PSNR delta over FUGR-C from no-reference features. On held-out test sequences,
it selects the candidate only when the predicted gain exceeds a margin; otherwise
it falls back to FUGR-C.

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
    return float(np.mean([
        np.mean(np.abs((xs[i] - xs[i - 1]) - (ys[i] - ys[i - 1])))
        for i in range(1, len(xs))
    ]))


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


def make_lambda(basic, fugr, a):
    return [np.clip(b + a * (f - b), 0, 1) for b, f in zip(basic, fugr)]


def make_atten(basic, fugr, tau=0.02, gmin=0.95, blur=0.8):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        nb = []
        if i > 0:
            nb.append(rs[i - 1])
        if i + 1 < len(rs):
            nb.append(rs[i + 1])
        if nb:
            avg = np.mean(np.stack(nb, axis=0), axis=0)
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
        nb = [r]
        if i > 0:
            nb.append(rs[i - 1])
        if i + 1 < len(rs):
            nb.append(rs[i + 1])
        med = np.median(np.stack(nb, axis=0), axis=0)
        outs.append(np.clip(basic[i] + (1 - beta) * r + beta * med, 0, 1))
    return outs


def make_clip(basic, fugr, scale=1.2):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        nb = [np.abs(r)]
        if i > 0:
            nb.append(np.abs(rs[i - 1]))
        if i + 1 < len(rs):
            nb.append(np.abs(rs[i + 1]))
        lim = scale * np.median(np.stack(nb, axis=0), axis=0) + 1e-4
        outs.append(np.clip(basic[i] + np.clip(r, -lim, lim), 0, 1))
    return outs


def build_candidates(blur):
    return [
        ("FUGR-C", lambda b, f: f),
        ("Lambda-0.85", lambda b, f: make_lambda(b, f, 0.85)),
        ("Lambda-0.95", lambda b, f: make_lambda(b, f, 0.95)),
        ("Lambda-1.05", lambda b, f: make_lambda(b, f, 1.05)),
        ("Lambda-1.20", lambda b, f: make_lambda(b, f, 1.20)),
        ("ResidualAtten-t0.020-g0.95", lambda b, f: make_atten(b, f, 0.020, 0.95, blur)),
        ("ResidualAtten-t0.010-g0.95", lambda b, f: make_atten(b, f, 0.010, 0.95, blur)),
        ("ResidualMedian-b0.25", lambda b, f: make_median(b, f, 0.25)),
        ("ResidualClip-c1.20", lambda b, f: make_clip(b, f, 1.20)),
    ]


def frame_features(basic, fugr, idx):
    b = basic[idx]
    f = fugr[idx]
    r = f - b
    feats = [
        1.0,
        idx / max(1, len(basic) - 1),
        float(np.mean(np.abs(r))),
        float(np.std(r)),
        float(np.max(np.abs(r))),
        sharpness(b),
        sharpness(f),
        sharpness(f) - sharpness(b),
        float(np.mean(np.abs(f - b))),
    ]

    for arr in [basic, fugr]:
        prev_diff = 0.0 if idx == 0 else float(np.mean(np.abs(arr[idx] - arr[idx - 1])))
        next_diff = 0.0 if idx + 1 >= len(arr) else float(np.mean(np.abs(arr[idx + 1] - arr[idx])))
        feats += [prev_diff, next_diff, 0.5 * (prev_diff + next_diff)]

    rs = [fu - ba for ba, fu in zip(basic, fugr)]
    prev_ri = 0.0 if idx == 0 else float(np.mean(np.abs(rs[idx] - rs[idx - 1])))
    next_ri = 0.0 if idx + 1 >= len(rs) else float(np.mean(np.abs(rs[idx + 1] - rs[idx])))
    feats += [prev_ri, next_ri, 0.5 * (prev_ri + next_ri)]
    return np.asarray(feats, dtype=np.float64)


def eval_sequence(selected, gt):
    return {
        "num_frames": len(selected),
        "psnr": float(np.mean([psnr(x, y) for x, y in zip(selected, gt)])),
        "ssim": float(np.mean([ssim_rgb(x, y) for x, y in zip(selected, gt)])),
        "sharpness": float(np.mean([sharpness(x) for x in selected])),
        "tde": tde(selected, gt),
    }


def ridge_fit(X, y, alpha):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_seqs", nargs="+", default=["000", "003", "006", "010", "011"])
    ap.add_argument("--test_seqs", nargs="+", default=["018", "020", "026", "028", "029"])
    ap.add_argument("--blur_sigma", type=float, default=0.8)
    ap.add_argument("--ridge_alpha", type=float, default=1e-3)
    ap.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.0001, 0.0005, 0.001, 0.002])
    args = ap.parse_args()

    out = Path(args.out_dir)
    md = out / "metrics"
    fd = out / "figures"
    md.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)

    data = collect(args.input_dir)
    candidates = build_candidates(args.blur_sigma)
    methods = [m for m, _ in candidates]
    nonfugr = [m for m in methods if m != "FUGR-C"]

    outputs = {}
    gt_by_seq = {}
    feats_by_seq_frame = {}
    frame_rows = []

    for seq, items in data.items():
        basic = [it["basic"] for it in items]
        fugr = [it["fugr"] for it in items]
        gt = [it["gt"] for it in items]
        gt_by_seq[seq] = gt
        outputs[seq] = {}
        for method, fn in candidates:
            outputs[seq][method] = fn(basic, fugr)

        for idx, it in enumerate(items):
            feats = frame_features(basic, fugr, idx)
            feats_by_seq_frame[(seq, idx)] = feats
            fugr_psnr = psnr(outputs[seq]["FUGR-C"][idx], gt[idx])
            for method in methods:
                p = psnr(outputs[seq][method][idx], gt[idx])
                frame_rows.append({
                    "sequence": seq,
                    "frame": it["frame"],
                    "frame_index": idx,
                    "method": method,
                    "psnr": p,
                    "delta_psnr_vs_fugr": p - fugr_psnr,
                })

    models = {}
    train_pred_rows = []
    for method in nonfugr:
        X, y = [], []
        for r in frame_rows:
            if r["sequence"] in args.calib_seqs and r["method"] == method:
                X.append(feats_by_seq_frame[(r["sequence"], int(r["frame_index"]))])
                y.append(float(r["delta_psnr_vs_fugr"]))
        w = ridge_fit(X, y, args.ridge_alpha)
        models[method] = w
        for xi, yi in zip(X, y):
            train_pred_rows.append({"method": method, "true_delta": yi, "pred_delta": float(np.dot(xi, w))})

    policy_summaries = []
    policy_details = []
    selection_rows = []

    def add_policy(policy, seq_to_selected):
        seq_eval_rows = []
        for seq in args.test_seqs:
            xs = seq_to_selected(seq)
            gt = gt_by_seq[seq]
            rec = eval_sequence(xs, gt)
            rec.update({"policy": policy, "sequence": seq})
            seq_eval_rows.append(rec)

            fr = eval_sequence(outputs[seq]["FUGR-C"], gt)
            policy_details.append({
                "policy": policy,
                "sequence": seq,
                "psnr": rec["psnr"],
                "tde": rec["tde"],
                "fugr_psnr": fr["psnr"],
                "fugr_tde": fr["tde"],
                "delta_psnr_vs_fugr": rec["psnr"] - fr["psnr"],
                "delta_tde_vs_fugr": rec["tde"] - fr["tde"],
            })

        agg = {
            "policy": policy,
            "num_sequences": len(seq_eval_rows),
            "num_frames": int(sum(r["num_frames"] for r in seq_eval_rows)),
            "psnr": float(np.mean([r["psnr"] for r in seq_eval_rows])),
            "ssim": float(np.mean([r["ssim"] for r in seq_eval_rows])),
            "sharpness": float(np.mean([r["sharpness"] for r in seq_eval_rows])),
            "tde": float(np.mean([r["tde"] for r in seq_eval_rows])),
        }
        fugr_rows = [eval_sequence(outputs[s]["FUGR-C"], gt_by_seq[s]) for s in args.test_seqs]
        fugr_agg = {
            "psnr": float(np.mean([r["psnr"] for r in fugr_rows])),
            "tde": float(np.mean([r["tde"] for r in fugr_rows])),
        }
        agg["delta_psnr_vs_fugr"] = agg["psnr"] - fugr_agg["psnr"]
        agg["delta_tde_vs_fugr"] = agg["tde"] - fugr_agg["tde"]
        policy_summaries.append(agg)

    add_policy("FUGR-C", lambda seq: outputs[seq]["FUGR-C"])
    for method in nonfugr:
        add_policy(f"Fixed-{method}", lambda seq, m=method: outputs[seq][m])

    for margin in args.margins:
        def selector(seq, margin=margin):
            selected = []
            for idx in range(len(gt_by_seq[seq])):
                feats = feats_by_seq_frame[(seq, idx)]
                preds = [(float(np.dot(feats, models[m])), m) for m in nonfugr]
                best_pred, best_method = max(preds, key=lambda z: z[0])
                chosen = best_method if best_pred > margin else "FUGR-C"
                selected.append(outputs[seq][chosen][idx])
                selection_rows.append({
                    "policy": f"LearnedSelector-margin{margin:.4f}",
                    "sequence": seq,
                    "frame_index": idx,
                    "chosen_method": chosen,
                    "pred_delta": best_pred,
                    "margin": margin,
                })
            return selected

        add_policy(f"LearnedSelector-margin{margin:.4f}", selector)

    def oracle_selector(seq, non_only=False):
        selected = []
        for idx in range(len(gt_by_seq[seq])):
            cand = nonfugr if non_only else methods
            best_method = max(cand, key=lambda m: psnr(outputs[seq][m][idx], gt_by_seq[seq][idx]))
            selected.append(outputs[seq][best_method][idx])
        return selected

    add_policy("TestPerFrameOracle", lambda seq: oracle_selector(seq, non_only=False))
    add_policy("TestPerFrameOracleNonFUGR", lambda seq: oracle_selector(seq, non_only=True))

    def write_csv(path, rows, fields):
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(md / "A9_frame_candidate_metrics.csv", frame_rows,
              ["sequence", "frame", "frame_index", "method", "psnr", "delta_psnr_vs_fugr"])
    write_csv(md / "A9_train_prediction_rows.csv", train_pred_rows,
              ["method", "true_delta", "pred_delta"])
    write_csv(md / "A9_selection_rows.csv", selection_rows,
              ["policy", "sequence", "frame_index", "chosen_method", "pred_delta", "margin"])
    write_csv(md / "A9_policy_summary.csv", policy_summaries,
              ["policy", "num_sequences", "num_frames", "psnr", "ssim", "sharpness", "tde",
               "delta_psnr_vs_fugr", "delta_tde_vs_fugr"])
    write_csv(md / "A9_policy_sequence_details.csv", policy_details,
              ["policy", "sequence", "psnr", "tde", "fugr_psnr", "fugr_tde",
               "delta_psnr_vs_fugr", "delta_tde_vs_fugr"])

    order = sorted(policy_summaries, key=lambda r: r["delta_psnr_vs_fugr"], reverse=True)
    plt.figure(figsize=(10, 4))
    plt.bar([r["policy"] for r in order], [r["delta_psnr_vs_fugr"] for r in order])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=75, ha="right", fontsize=6)
    plt.ylabel("PSNR delta vs FUGR-C")
    plt.title("A9 learned no-reference selector")
    plt.tight_layout()
    plt.savefig(fd / "A9_policy_delta_psnr.png", dpi=300)
    plt.close()

    non_oracle = [r for r in policy_summaries if "Oracle" not in r["policy"]]
    best_non_oracle = max(non_oracle, key=lambda r: r["psnr"])
    oracle = next(r for r in policy_summaries if r["policy"] == "TestPerFrameOracle")
    fugr = next(r for r in policy_summaries if r["policy"] == "FUGR-C")

    txt = md / "A9_learned_selector_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A9: Learned No-reference Frame-level Selector\n\n")
        f.write("FUGR-C test reference:\n")
        f.write(str(fugr) + "\n\n")
        f.write("Best non-oracle deployable policy:\n")
        f.write(str(best_non_oracle) + "\n\n")
        f.write("Test per-frame oracle:\n")
        f.write(str(oracle) + "\n\n")
        f.write("Interpretation:\n")
        f.write(
            "A9 trains a small ridge-regression selector on calibration frames to predict which "
            "non-FUGR candidate may improve PSNR over FUGR-C using only no-reference features. "
            "If the learned selector cannot beat FUGR-C on held-out test sequences, then the "
            "remaining Direction-A headroom is not only small but also hard to exploit even with "
            "a supervised calibration stage.\n"
        )

    print(txt.read_text())
    print("Saved:", out)


if __name__ == "__main__":
    main()
