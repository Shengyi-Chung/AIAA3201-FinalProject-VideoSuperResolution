#!/usr/bin/env python3
"""
Direction A8: no-reference adaptive policy exploration.

Motivation:
  A1-A7 show that fixed temporal post-processing variants cannot beat FUGR-C.
  A natural remaining question is whether an adaptive policy can select or blend
  temporal candidates according to sequence-level no-reference features.

This script tests several deployable no-GT policies selected on calibration sequences
and evaluated on held-out test sequences:

  Policy 1: global best non-FUGR method on calibration.
  Policy 2: motion-bucket best non-FUGR method.
  Policy 3: residual-instability-bucket best non-FUGR method.
  Policy 4: risk-threshold fallback:
      use best non-FUGR candidate only when no-reference features predict a safer case;
      otherwise use FUGR-C.

It also reports oracle references:
  - per-sequence oracle using GT;
  - per-frame oracle using GT.

Input:
  input_dir/frames/<seq>/<frame>_basic.png
  input_dir/frames/<seq>/<frame>_fugr.png
  input_dir/frames/<seq>/<frame>_gt.png

Expected use:
  --input_dir $OUT/DirectionB/expanded_strong_st015
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
    return float(np.mean([np.mean(np.abs((xs[i] - xs[i-1]) - (ys[i] - ys[i-1])))
                          for i in range(1, len(xs))]))


def no_ref_motion(xs):
    if len(xs) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(xs[i] - xs[i-1])) for i in range(1, len(xs))]))


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


def residual_instability(rs):
    if len(rs) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(rs[i] - rs[i-1])) for i in range(1, len(rs))]))


def residual_energy(rs):
    return float(np.mean([np.mean(np.abs(r)) for r in rs]))


def make_lambda(basic, fugr, lam):
    return [np.clip(b + lam * (f - b), 0, 1) for b, f in zip(basic, fugr)]


def make_atten(basic, fugr, tau, gmin, blur):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        neigh = []
        if i > 0:
            neigh.append(rs[i-1])
        if i + 1 < len(rs):
            neigh.append(rs[i+1])
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


def make_median(basic, fugr, beta):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        neigh = [r]
        if i > 0:
            neigh.append(rs[i-1])
        if i + 1 < len(rs):
            neigh.append(rs[i+1])
        med = np.median(np.stack(neigh, axis=0), axis=0)
        outs.append(np.clip(basic[i] + (1 - beta) * r + beta * med, 0, 1))
    return outs


def make_clip(basic, fugr, scale):
    rs = residuals(basic, fugr)
    outs = []
    for i, r in enumerate(rs):
        neigh = [np.abs(r)]
        if i > 0:
            neigh.append(np.abs(rs[i-1]))
        if i + 1 < len(rs):
            neigh.append(np.abs(rs[i+1]))
        limit = scale * np.median(np.stack(neigh, axis=0), axis=0) + 1e-4
        outs.append(np.clip(basic[i] + np.clip(r, -limit, limit), 0, 1))
    return outs


def build_candidates(blur):
    return [
        ("FUGR-C", lambda b, f: f),
        ("BasicVSR", lambda b, f: b),
        ("Lambda-0.85", lambda b, f: make_lambda(b, f, 0.85)),
        ("Lambda-0.95", lambda b, f: make_lambda(b, f, 0.95)),
        ("Lambda-1.05", lambda b, f: make_lambda(b, f, 1.05)),
        ("Lambda-1.20", lambda b, f: make_lambda(b, f, 1.20)),
        ("ResidualAtten-t0.020-g0.95", lambda b, f: make_atten(b, f, 0.020, 0.95, blur)),
        ("ResidualAtten-t0.010-g0.95", lambda b, f: make_atten(b, f, 0.010, 0.95, blur)),
        ("ResidualMedian-b0.25", lambda b, f: make_median(b, f, 0.25)),
        ("ResidualClip-c1.20", lambda b, f: make_clip(b, f, 1.20)),
    ]


def eval_outputs(xs, ys):
    return {
        "num_frames": len(xs),
        "psnr": float(np.mean([psnr(x, y) for x, y in zip(xs, ys)])),
        "ssim": float(np.mean([ssim_rgb(x, y) for x, y in zip(xs, ys)])),
        "sharpness": float(np.mean([sharpness(x) for x in xs])),
        "tde": tde(xs, ys),
    }


def aggregate(seq_rows, name):
    return {
        "policy": name,
        "num_sequences": len(seq_rows),
        "num_frames": int(sum(r["num_frames"] for r in seq_rows)),
        "psnr": float(np.mean([r["psnr"] for r in seq_rows])),
        "ssim": float(np.mean([r["ssim"] for r in seq_rows])),
        "sharpness": float(np.mean([r["sharpness"] for r in seq_rows])),
        "tde": float(np.mean([r["tde"] for r in seq_rows])),
    }


def bucket_by_quantile(values, x):
    q1, q2 = np.quantile(values, [1/3, 2/3])
    if x <= q1:
        return "low"
    if x <= q2:
        return "mid"
    return "high"


def choose_best_method(rows, allowed_methods, criterion, psnr_loss_limit, fugr_psnr=None):
    cand = [r for r in rows if r["method"] in allowed_methods]
    if fugr_psnr is not None:
        cand = [r for r in cand if r["psnr"] >= fugr_psnr - psnr_loss_limit]
    if criterion == "tde":
        return sorted(cand, key=lambda r: (r["tde"], -r["psnr"]))[0]["method"]
    if criterion == "psnr":
        return sorted(cand, key=lambda r: (r["psnr"], -r["tde"]), reverse=True)[0]["method"]
    raise ValueError(criterion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_seqs", nargs="+", default=["000", "003", "006", "010", "011"])
    ap.add_argument("--test_seqs", nargs="+", default=["018", "020", "026", "028", "029"])
    ap.add_argument("--blur_sigma", type=float, default=0.8)
    ap.add_argument("--psnr_loss_limit", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.out_dir)
    md = out / "metrics"
    fd = out / "figures"
    md.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)

    data = collect(args.input_dir)
    candidates = build_candidates(args.blur_sigma)
    methods = [m for m, _ in candidates]

    # Precompute outputs / metrics / features.
    outputs = {}
    seq_method_rows = []
    features = []

    for seq, items in data.items():
        basic = [it["basic"] for it in items]
        fugr = [it["fugr"] for it in items]
        gt = [it["gt"] for it in items]
        rs = residuals(basic, fugr)

        outputs[seq] = {}
        feat = {
            "sequence": seq,
            "no_ref_motion_fugr": no_ref_motion(fugr),
            "no_ref_motion_basic": no_ref_motion(basic),
            "residual_energy": residual_energy(rs),
            "residual_instability": residual_instability(rs),
        }
        features.append(feat)

        for method, fn in candidates:
            xs = fn(basic, fugr)
            outputs[seq][method] = xs
            rec = eval_outputs(xs, gt)
            rec.update({"sequence": seq, "method": method})
            seq_method_rows.append(rec)

    feat_by_seq = {f["sequence"]: f for f in features}
    rows_by_seq_method = {(r["sequence"], r["method"]): r for r in seq_method_rows}

    def eval_policy(policy_name, seq_to_method, seqs):
        rows = []
        details = []
        for seq in seqs:
            m = seq_to_method(seq)
            row = dict(rows_by_seq_method[(seq, m)])
            rows.append(row)
            fr = rows_by_seq_method[(seq, "FUGR-C")]
            details.append({
                "policy": policy_name,
                "sequence": seq,
                "selected_method": m,
                "psnr": row["psnr"],
                "tde": row["tde"],
                "fugr_psnr": fr["psnr"],
                "fugr_tde": fr["tde"],
                "delta_psnr_vs_fugr": row["psnr"] - fr["psnr"],
                "delta_tde_vs_fugr": row["tde"] - fr["tde"],
            })
        agg = aggregate(rows, policy_name)
        fugr_agg = aggregate([rows_by_seq_method[(s, "FUGR-C")] for s in seqs], "FUGR-C")
        agg["delta_psnr_vs_fugr"] = agg["psnr"] - fugr_agg["psnr"]
        agg["delta_tde_vs_fugr"] = agg["tde"] - fugr_agg["tde"]
        return agg, details

    calib_rows = [r for r in seq_method_rows if r["sequence"] in args.calib_seqs]
    test_seqs = args.test_seqs

    # Baselines.
    policy_summaries = []
    policy_details = []

    for baseline in ["FUGR-C", "BasicVSR"]:
        agg, det = eval_policy(baseline, lambda seq, m=baseline: m, test_seqs)
        policy_summaries.append(agg)
        policy_details += det

    nonfugr_methods = [m for m in methods if m != "FUGR-C"]

    # Policy 1: global best on calibration.
    fugr_calib = aggregate([rows_by_seq_method[(s, "FUGR-C")] for s in args.calib_seqs], "FUGR-C")
    global_non = choose_best_method(calib_rows, nonfugr_methods, "tde", args.psnr_loss_limit, fugr_calib["psnr"])
    agg, det = eval_policy(f"GlobalCalibNonFUGR-{global_non}", lambda seq, m=global_non: m, test_seqs)
    policy_summaries.append(agg)
    policy_details += det

    # Policy 2/3: bucket policies.
    for feature_key, policy_prefix in [
        ("no_ref_motion_fugr", "MotionBucket"),
        ("residual_instability", "ResidualInstabilityBucket"),
        ("residual_energy", "ResidualEnergyBucket"),
    ]:
        calib_vals = [feat_by_seq[s][feature_key] for s in args.calib_seqs]
        calib_bucket_by_seq = {s: bucket_by_quantile(calib_vals, feat_by_seq[s][feature_key]) for s in args.calib_seqs}

        bucket_method = {}
        for bucket in ["low", "mid", "high"]:
            seqs_in_bucket = [s for s in args.calib_seqs if calib_bucket_by_seq[s] == bucket]
            if not seqs_in_bucket:
                bucket_method[bucket] = global_non
                continue
            rows = [r for r in calib_rows if r["sequence"] in seqs_in_bucket]
            fugr_rows = [rows_by_seq_method[(s, "FUGR-C")] for s in seqs_in_bucket]
            fugr_bucket = aggregate(fugr_rows, "FUGR-C")
            bucket_method[bucket] = choose_best_method(rows, nonfugr_methods, "tde", args.psnr_loss_limit, fugr_bucket["psnr"])

        # Apply calibration quantiles to test.
        def seq_to_method(seq, fk=feature_key, bm=bucket_method, vals=calib_vals):
            return bm[bucket_by_quantile(vals, feat_by_seq[seq][fk])]

        agg, det = eval_policy(f"{policy_prefix}Adaptive", seq_to_method, test_seqs)
        policy_summaries.append(agg)
        policy_details += det

    # Policy 4: fallback policy, using non-FUGR only for calibration-positive feature ranges.
    # Find sequences in calibration where global_non improves PSNR over FUGR, then derive a conservative
    # residual instability threshold. If no positive sequence exists, this policy falls back to FUGR-C.
    positive_calib = []
    for s in args.calib_seqs:
        r = rows_by_seq_method[(s, global_non)]
        fr = rows_by_seq_method[(s, "FUGR-C")]
        if r["psnr"] > fr["psnr"]:
            positive_calib.append(s)

    if positive_calib:
        max_instab = max(feat_by_seq[s]["residual_instability"] for s in positive_calib)
        max_energy = max(feat_by_seq[s]["residual_energy"] for s in positive_calib)

        def fallback(seq):
            f = feat_by_seq[seq]
            if f["residual_instability"] <= max_instab and f["residual_energy"] <= max_energy:
                return global_non
            return "FUGR-C"

        pname = f"Fallback-{global_non}"
    else:
        def fallback(seq):
            return "FUGR-C"
        pname = "Fallback-no-positive-calib-use-FUGR"

    agg, det = eval_policy(pname, fallback, test_seqs)
    policy_summaries.append(agg)
    policy_details += det

    # Oracle references on test only.
    oracle_details = []
    oracle_rows = []
    oracle_non_rows = []
    for s in test_seqs:
        gt = [it["gt"] for it in data[s]]
        # per-sequence oracle
        best_seq = max([rows_by_seq_method[(s, m)] for m in methods], key=lambda r: r["psnr"])
        best_non_seq = max([rows_by_seq_method[(s, m)] for m in nonfugr_methods], key=lambda r: r["psnr"])

        # per-frame oracle
        xs, xs_non = [], []
        for i in range(len(gt)):
            scores = [(psnr(outputs[s][m][i], gt[i]), m) for m in methods]
            scores_non = [(psnr(outputs[s][m][i], gt[i]), m) for m in nonfugr_methods]
            bm = max(scores)[1]
            bmn = max(scores_non)[1]
            xs.append(outputs[s][bm][i])
            xs_non.append(outputs[s][bmn][i])

        orow = eval_outputs(xs, gt)
        orow.update({"sequence": s, "method": "PerFrameOracle"})
        oracle_rows.append(orow)

        onrow = eval_outputs(xs_non, gt)
        onrow.update({"sequence": s, "method": "PerFrameOracleNonFUGR"})
        oracle_non_rows.append(onrow)

        fr = rows_by_seq_method[(s, "FUGR-C")]
        oracle_details.append({
            "sequence": s,
            "best_seq_method": best_seq["method"],
            "best_seq_delta_psnr_vs_fugr": best_seq["psnr"] - fr["psnr"],
            "best_non_seq_method": best_non_seq["method"],
            "best_non_seq_delta_psnr_vs_fugr": best_non_seq["psnr"] - fr["psnr"],
        })

    for label, rows in [("TestPerFrameOracle", oracle_rows), ("TestPerFrameOracleNonFUGR", oracle_non_rows)]:
        agg = aggregate(rows, label)
        fugr_agg = aggregate([rows_by_seq_method[(s, "FUGR-C")] for s in test_seqs], "FUGR-C")
        agg["delta_psnr_vs_fugr"] = agg["psnr"] - fugr_agg["psnr"]
        agg["delta_tde_vs_fugr"] = agg["tde"] - fugr_agg["tde"]
        policy_summaries.append(agg)

    # Save files.
    def write_csv(path, rows, fields):
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(md / "A8_sequence_features.csv", features,
              ["sequence", "no_ref_motion_fugr", "no_ref_motion_basic", "residual_energy", "residual_instability"])

    write_csv(md / "A8_seq_method_metrics.csv", seq_method_rows,
              ["sequence", "method", "num_frames", "psnr", "ssim", "sharpness", "tde"])

    write_csv(md / "A8_policy_summary.csv", policy_summaries,
              ["policy", "num_sequences", "num_frames", "psnr", "ssim", "sharpness", "tde",
               "delta_psnr_vs_fugr", "delta_tde_vs_fugr"])

    write_csv(md / "A8_policy_sequence_details.csv", policy_details,
              ["policy", "sequence", "selected_method", "psnr", "tde", "fugr_psnr", "fugr_tde",
               "delta_psnr_vs_fugr", "delta_tde_vs_fugr"])

    write_csv(md / "A8_oracle_test_details.csv", oracle_details,
              ["sequence", "best_seq_method", "best_seq_delta_psnr_vs_fugr",
               "best_non_seq_method", "best_non_seq_delta_psnr_vs_fugr"])

    # Figure.
    plt.figure(figsize=(8, 4))
    order = sorted(policy_summaries, key=lambda r: r["delta_psnr_vs_fugr"], reverse=True)
    plt.bar([r["policy"] for r in order], [r["delta_psnr_vs_fugr"] for r in order])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.ylabel("PSNR delta vs FUGR-C")
    plt.title("A8 deployable policies vs oracle references")
    plt.tight_layout()
    plt.savefig(fd / "A8_policy_delta_psnr.png", dpi=300)
    plt.close()

    best_policy = max([r for r in policy_summaries if not r["policy"].startswith("TestPerFrameOracle")],
                      key=lambda r: r["psnr"])
    fugr_test = next(r for r in policy_summaries if r["policy"] == "FUGR-C")
    oracle_test = next(r for r in policy_summaries if r["policy"] == "TestPerFrameOracle")

    txt = md / "A8_adaptive_policy_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A8: No-reference Adaptive Policy Exploration\n\n")
        f.write("FUGR-C test reference:\n")
        f.write(str(fugr_test) + "\n\n")
        f.write("Best deployable policy on test:\n")
        f.write(str(best_policy) + "\n\n")
        f.write("Oracle reference on test:\n")
        f.write(str(oracle_test) + "\n\n")
        f.write("Selected global non-FUGR candidate on calibration:\n")
        f.write(global_non + "\n\n")
        f.write("Interpretation:\n")
        f.write(
            "A8 tests whether no-reference sequence-level features can select a better temporal refinement candidate. "
            "If deployable policies do not beat FUGR-C while oracle references remain tiny, this suggests that the "
            "remaining A-direction headroom is too small and too unstable to be exploited reliably without ground truth.\n"
        )

    print(txt.read_text())
    print("Saved:", out)


if __name__ == "__main__":
    main()
