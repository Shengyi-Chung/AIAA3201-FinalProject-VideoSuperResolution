#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))

def tde(xs, ys):
    if len(xs) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs((xs[i] - xs[i-1]) - (ys[i] - ys[i-1]))) for i in range(1, len(xs))]))

def motion(ys):
    if len(ys) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(ys[i] - ys[i-1])) for i in range(1, len(ys))]))

def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def collect(input_dir):
    root = Path(input_dir) / "frames"
    data = {}
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        items = []
        for fp in sorted(sd.glob("*_fugr.png")):
            frame = fp.name.replace("_fugr.png", "")
            bp, gp = sd / f"{frame}_basic.png", sd / f"{frame}_gt.png"
            if bp.exists() and gp.exists():
                items.append({"frame": frame, "basic": read_rgb(bp), "fugr": read_rgb(fp), "gt": read_rgb(gp)})
        if items:
            data[sd.name] = items
    return data

def residuals(basic, fugr):
    return [f - b for b, f in zip(basic, fugr)]

def lam(basic, fugr, a):
    return [np.clip(b + a * (f - b), 0, 1) for b, f in zip(basic, fugr)]

def rmedian(basic, fugr, beta):
    r = residuals(basic, fugr); outs = []
    for i in range(len(r)):
        nb = [r[i]]
        if i > 0: nb.append(r[i-1])
        if i+1 < len(r): nb.append(r[i+1])
        med = np.median(np.stack(nb, 0), 0)
        outs.append(np.clip(basic[i] + (1-beta)*r[i] + beta*med, 0, 1))
    return outs

def ratten(basic, fugr, tau=0.02, gmin=0.95, blur=0.8):
    r = residuals(basic, fugr); outs = []
    for i in range(len(r)):
        nb = []
        if i > 0: nb.append(r[i-1])
        if i+1 < len(r): nb.append(r[i+1])
        if nb:
            risk = np.mean(np.abs(r[i] - np.mean(np.stack(nb, 0), 0)), axis=2)
            gamma = gmin + (1-gmin) * np.exp(-risk / tau)
            if blur > 0:
                gamma = cv2.GaussianBlur(gamma, (0, 0), blur)
            gamma = np.clip(gamma, gmin, 1.0)
        else:
            gamma = np.ones(r[i].shape[:2], dtype=np.float32)
        outs.append(np.clip(basic[i] + gamma[..., None] * r[i], 0, 1))
    return outs

def rclip(basic, fugr, scale=1.2):
    r = residuals(basic, fugr); outs = []
    for i in range(len(r)):
        nb = [np.abs(r[i])]
        if i > 0: nb.append(np.abs(r[i-1]))
        if i+1 < len(r): nb.append(np.abs(r[i+1]))
        lim = scale * np.median(np.stack(nb, 0), 0) + 1e-4
        outs.append(np.clip(basic[i] + np.clip(r[i], -lim, lim), 0, 1))
    return outs

def eval_seq(xs, gt):
    return {
        "num_frames": len(xs),
        "psnr": float(np.mean([psnr(x, y) for x, y in zip(xs, gt)])),
        "tde": tde(xs, gt),
        "motion": motion(gt)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--blur_sigma", type=float, default=0.8)
    args = ap.parse_args()

    out = Path(args.out_dir); md = out / "metrics"; fd = out / "figures"
    md.mkdir(parents=True, exist_ok=True); fd.mkdir(parents=True, exist_ok=True)
    data = collect(args.input_dir)
    seqs = sorted(data)
    methods = [
        ("BasicVSR", lambda b, f: b),
        ("FUGR-C", lambda b, f: f),
        ("Lambda-0.85", lambda b, f: lam(b, f, 0.85)),
        ("Lambda-0.95", lambda b, f: lam(b, f, 0.95)),
        ("Lambda-1.05", lambda b, f: lam(b, f, 1.05)),
        ("Lambda-1.20", lambda b, f: lam(b, f, 1.20)),
        ("ResidualMedian-b0.25", lambda b, f: rmedian(b, f, 0.25)),
        ("ResidualAtten-t0.020-g0.95", lambda b, f: ratten(b, f, 0.02, 0.95, args.blur_sigma)),
        ("ResidualClip-c1.20", lambda b, f: rclip(b, f, 1.2)),
    ]

    seq_rows, frame_rows, features = [], [], []
    outputs = {}
    for seq in seqs:
        items = data[seq]
        basic = [x["basic"] for x in items]
        fugr = [x["fugr"] for x in items]
        gt = [x["gt"] for x in items]
        frames = [x["frame"] for x in items]
        res = residuals(basic, fugr)

        bm, fm = eval_seq(basic, gt), eval_seq(fugr, gt)
        instab = float(np.mean([np.mean(np.abs(res[i]-res[i-1])) for i in range(1, len(res))])) if len(res) > 1 else 0.0
        features.append({
            "sequence": seq,
            "motion": motion(gt),
            "residual_energy": float(np.mean([np.mean(np.abs(r)) for r in res])),
            "residual_temporal_instability": instab,
            "fugr_delta_psnr_vs_basic": fm["psnr"] - bm["psnr"],
            "fugr_delta_tde_vs_basic": fm["tde"] - bm["tde"],
        })

        outputs[seq] = {}
        for name, fn in methods:
            xs = fn(basic, fugr)
            outputs[seq][name] = xs
            rec = eval_seq(xs, gt); rec.update({"sequence": seq, "method": name})
            seq_rows.append(rec)
            for frame, x, y in zip(frames, xs, gt):
                frame_rows.append({"sequence": seq, "frame": frame, "method": name, "psnr": psnr(x, y)})

    # global summary
    summary = []
    for name, _ in methods:
        rs = [r for r in seq_rows if r["method"] == name]
        summary.append({
            "method": name,
            "num_sequences": len(rs),
            "num_frames": sum(r["num_frames"] for r in rs),
            "motion": float(np.mean([r["motion"] for r in rs])),
            "psnr": float(np.mean([r["psnr"] for r in rs])),
            "tde": float(np.mean([r["tde"] for r in rs])),
        })
    fugr = next(r for r in summary if r["method"] == "FUGR-C")
    for r in summary:
        r["delta_psnr_vs_fugr"] = r["psnr"] - fugr["psnr"]
        r["delta_tde_vs_fugr"] = r["tde"] - fugr["tde"]

    # per-frame oracle
    oracle_frames = []
    oracle_seq_rows, oracle_non_seq_rows = [], []
    for seq in seqs:
        items = data[seq]; gt = [x["gt"] for x in items]; frames = [x["frame"] for x in items]
        oxs, onxs = [], []
        for i, (frame, y) in enumerate(zip(frames, gt)):
            scores = [(psnr(outputs[seq][name][i], y), name, outputs[seq][name][i]) for name, _ in methods]
            scores_non = [s for s in scores if s[1] != "FUGR-C"]
            best = max(scores, key=lambda z: z[0])
            best_non = max(scores_non, key=lambda z: z[0])
            fp = psnr(outputs[seq]["FUGR-C"][i], y)
            oracle_frames.append({
                "sequence": seq, "frame": frame,
                "oracle_method": best[1], "oracle_delta_psnr_vs_fugr": best[0] - fp,
                "oracle_nonfugr_method": best_non[1], "oracle_nonfugr_delta_psnr_vs_fugr": best_non[0] - fp,
            })
            oxs.append(best[2]); onxs.append(best_non[2])
        rec = eval_seq(oxs, gt); rec.update({"sequence": seq, "method": "PerFrameOracle"})
        oracle_seq_rows.append(rec)
        rec = eval_seq(onxs, gt); rec.update({"sequence": seq, "method": "PerFrameOracleNonFUGR"})
        oracle_non_seq_rows.append(rec)

    oracle_summary = []
    for label, rows in [("PerFrameOracle", oracle_seq_rows), ("PerFrameOracleNonFUGR", oracle_non_seq_rows)]:
        oracle_summary.append({
            "method": label,
            "num_sequences": len(rows),
            "num_frames": sum(r["num_frames"] for r in rows),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "tde": float(np.mean([r["tde"] for r in rows])),
            "delta_psnr_vs_fugr": float(np.mean([r["psnr"] for r in rows])) - fugr["psnr"],
            "delta_tde_vs_fugr": float(np.mean([r["tde"] for r in rows])) - fugr["tde"],
        })

    # per-sequence oracle
    seq_oracle = []
    for seq in seqs:
        rows = [r for r in seq_rows if r["sequence"] == seq]
        fr = next(r for r in rows if r["method"] == "FUGR-C")
        non = [r for r in rows if r["method"] != "FUGR-C"]
        bp = max(rows, key=lambda r: r["psnr"])
        bt = min(rows, key=lambda r: r["tde"])
        bnp = max(non, key=lambda r: r["psnr"])
        bnt = min(non, key=lambda r: r["tde"])
        seq_oracle.append({
            "sequence": seq,
            "fugr_psnr": fr["psnr"], "fugr_tde": fr["tde"],
            "best_psnr_method": bp["method"], "best_psnr_delta_vs_fugr": bp["psnr"] - fr["psnr"],
            "best_tde_method": bt["method"], "best_tde_delta_vs_fugr": bt["tde"] - fr["tde"],
            "best_nonfugr_psnr_method": bnp["method"], "best_nonfugr_psnr_delta_vs_fugr": bnp["psnr"] - fr["psnr"],
            "best_nonfugr_tde_method": bnt["method"], "best_nonfugr_tde_delta_vs_fugr": bnt["tde"] - fr["tde"],
        })

    # proxy correlations
    proxy_rows = []
    feat = {f["sequence"]: f for f in features}
    for name, _ in methods:
        if name == "FUGR-C": continue
        deltas, mot, ene, ins, fgain = [], [], [], [], []
        for seq in seqs:
            r = next(x for x in seq_rows if x["sequence"] == seq and x["method"] == name)
            fr = next(x for x in seq_rows if x["sequence"] == seq and x["method"] == "FUGR-C")
            deltas.append(r["psnr"] - fr["psnr"])
            mot.append(feat[seq]["motion"])
            ene.append(feat[seq]["residual_energy"])
            ins.append(feat[seq]["residual_temporal_instability"])
            fgain.append(feat[seq]["fugr_delta_psnr_vs_basic"])
        proxy_rows.append({
            "method": name,
            "mean_delta_psnr_vs_fugr": float(np.mean(deltas)),
            "positive_sequences": int(np.sum(np.asarray(deltas) > 0)),
            "corr_with_motion": corr(deltas, mot),
            "corr_with_residual_energy": corr(deltas, ene),
            "corr_with_residual_instability": corr(deltas, ins),
            "corr_with_fugr_basic_gain": corr(deltas, fgain),
        })

    def write_csv(path, rows, fields):
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    write_csv(md/"A7_method_summary.csv", sorted(summary, key=lambda r:(r["tde"], -r["psnr"])),
              ["method","num_sequences","num_frames","motion","psnr","tde","delta_psnr_vs_fugr","delta_tde_vs_fugr"])
    write_csv(md/"A7_sequence_method_metrics.csv", seq_rows,
              ["sequence","method","num_frames","motion","psnr","tde"])
    write_csv(md/"A7_sequence_features.csv", features,
              ["sequence","motion","residual_energy","residual_temporal_instability","fugr_delta_psnr_vs_basic","fugr_delta_tde_vs_basic"])
    write_csv(md/"A7_per_frame_oracle.csv", oracle_frames,
              ["sequence","frame","oracle_method","oracle_delta_psnr_vs_fugr","oracle_nonfugr_method","oracle_nonfugr_delta_psnr_vs_fugr"])
    write_csv(md/"A7_oracle_summary.csv", oracle_summary,
              ["method","num_sequences","num_frames","psnr","tde","delta_psnr_vs_fugr","delta_tde_vs_fugr"])
    write_csv(md/"A7_per_sequence_oracle.csv", seq_oracle,
              ["sequence","fugr_psnr","fugr_tde","best_psnr_method","best_psnr_delta_vs_fugr","best_tde_method","best_tde_delta_vs_fugr","best_nonfugr_psnr_method","best_nonfugr_psnr_delta_vs_fugr","best_nonfugr_tde_method","best_nonfugr_tde_delta_vs_fugr"])
    write_csv(md/"A7_proxy_correlation.csv", sorted(proxy_rows, key=lambda r:r["mean_delta_psnr_vs_fugr"], reverse=True),
              ["method","mean_delta_psnr_vs_fugr","positive_sequences","corr_with_motion","corr_with_residual_energy","corr_with_residual_instability","corr_with_fugr_basic_gain"])

    # figures
    sm = sorted(summary, key=lambda r:(r["tde"], -r["psnr"]))
    plt.figure(figsize=(7,5))
    plt.scatter([r["tde"] for r in sm], [r["psnr"] for r in sm], s=30)
    plt.scatter([fugr["tde"]], [fugr["psnr"]], s=100, marker="*")
    for r in sm[:8]:
        plt.text(r["tde"], r["psnr"], r["method"].replace("Residual","R"), fontsize=6)
    plt.xlabel("TDE lower is better"); plt.ylabel("PSNR higher is better")
    plt.title("A7 diagnostic candidate tradeoff")
    plt.tight_layout(); plt.savefig(fd/"A7_candidate_pareto.png", dpi=300); plt.close()

    winners = {}
    for r in oracle_frames:
        winners[r["oracle_method"]] = winners.get(r["oracle_method"], 0) + 1
    keys = sorted(winners, key=lambda k:winners[k], reverse=True)
    plt.figure(figsize=(8,4))
    plt.bar(keys, [winners[k] for k in keys])
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.ylabel("Frame count"); plt.title("A7 per-frame oracle winners")
    plt.tight_layout(); plt.savefig(fd/"A7_oracle_winner_counts.png", dpi=300); plt.close()

    pos = [r for r in oracle_frames if r["oracle_delta_psnr_vs_fugr"] > 0]
    pos_non = [r for r in oracle_frames if r["oracle_nonfugr_delta_psnr_vs_fugr"] > 0]
    pos_non_001 = [r for r in oracle_frames if r["oracle_nonfugr_delta_psnr_vs_fugr"] > 0.01]
    best_non = min([r for r in summary if r["method"] != "FUGR-C"], key=lambda r:(r["tde"], -r["psnr"]))

    txt = md/"A7_diagnostic_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A7: Diagnostic / Oracle / Motion-stratified Analysis\n\n")
        f.write("FUGR-C reference:\n")
        f.write(str(fugr) + "\n\n")
        f.write("Best non-FUGR candidate by TDE:\n")
        f.write(str(best_non) + "\n\n")
        f.write("Oracle summary:\n")
        for r in oracle_summary: f.write(str(r) + "\n")
        f.write("\nPer-frame oracle headroom:\n")
        f.write(f"frames with any candidate better than FUGR-C: {len(pos)}/{len(oracle_frames)}\n")
        f.write(f"frames with non-FUGR candidate better than FUGR-C: {len(pos_non)}/{len(oracle_frames)}\n")
        f.write(f"frames with non-FUGR gain > 0.01 dB: {len(pos_non_001)}/{len(oracle_frames)}\n\n")
        f.write("Interpretation:\n")
        f.write("A7 checks whether Direction A failed because of missing hidden headroom. "
                "If oracle gains are tiny and proxy correlations are weak, then simple adaptive temporal selection is unlikely to outperform FUGR-C reliably.\n")
    print(txt.read_text())
    print("Saved:", out)

if __name__ == "__main__":
    main()
