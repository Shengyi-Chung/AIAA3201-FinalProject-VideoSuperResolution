#!/usr/bin/env python3
"""
Direction A11: Train-time temporal fine-tuning with a lightweight residual temporal refiner.

Purpose
-------
Previous A-direction experiments tested post-hoc temporal filtering, oracle selection,
adaptive policies, and calibrated linear residual ensembles. A11 moves beyond fixed
post-processing by training a small neural temporal refiner on calibration sequences.

Important:
  - GT is used only for calibration/train sequences.
  - Held-out test sequences are never used for training.
  - The refiner predicts a small correction on top of FUGR-C.
  - Loss = reconstruction loss + lambda_temporal * temporal-difference loss
           + lambda_reg * correction magnitude regularization.

Input format:
  input_dir/frames/<seq>/<frame>_basic.png
  input_dir/frames/<seq>/<frame>_fugr.png
  input_dir/frames/<seq>/<frame>_gt.png

Recommended input:
  $OUT/DirectionB/expanded_strong_st015

Example:
  python -u directionA11_train_temporal_refiner.py \
    --input_dir $OUT/DirectionB/expanded_strong_st015 \
    --out_dir $OUT/DirectionA/A11_temporal_refiner \
    --train_seqs 000 003 006 010 011 \
    --test_seqs 018 020 026 028 029 \
    --temporal_weights 0.00 0.02 0.05 0.10 0.20 \
    --steps 1500 \
    --patch_size 96 \
    --batch_size 8 \
    --lr 1e-4 \
    --max_corr 0.04 \
    --amp
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    g = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8),
                     cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return float(np.var(cv2.Laplacian(g, cv2.CV_32F)))


def tde(xs, ys):
    if len(xs) < 2:
        return 0.0
    vals = []
    for i in range(1, len(xs)):
        vals.append(float(np.mean(np.abs((xs[i] - xs[i - 1]) - (ys[i] - ys[i - 1])))))
    return float(np.mean(vals))


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


def to_torch_img(x, device):
    # HWC [0,1] -> CHW tensor
    return torch.from_numpy(x.transpose(2, 0, 1)).float().to(device)


def make_input_tensor(seq_items, idx, device):
    n = len(seq_items)
    ip = max(0, idx - 1)
    inx = min(n - 1, idx + 1)

    prev_f = seq_items[ip]["fugr"]
    cur_f = seq_items[idx]["fugr"]
    next_f = seq_items[inx]["fugr"]
    cur_b = seq_items[idx]["basic"]

    arr = np.concatenate([prev_f, cur_f, next_f, cur_b], axis=2)
    return to_torch_img(arr, device)


class TinyTemporalRefiner(nn.Module):
    def __init__(self, in_ch=12, hidden=48, max_corr=0.04):
        super().__init__()
        self.max_corr = float(max_corr)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

        # Start near identity/FUGR-C.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # x channels: prev_fugr, cur_fugr, next_fugr, cur_basic
        cur = x[:, 3:6, :, :]
        corr = self.max_corr * torch.tanh(self.net(x))
        out = torch.clamp(cur + corr, 0.0, 1.0)
        return out, corr


def charbonnier(x, eps=1e-3):
    return torch.mean(torch.sqrt(x * x + eps * eps))


def sample_pair_batch(data, train_seqs, batch_size, patch_size, device):
    xs_cur, xs_prev, gt_cur, gt_prev = [], [], [], []

    for _ in range(batch_size):
        seq = random.choice(train_seqs)
        items = data[seq]
        if len(items) < 2:
            raise ValueError(f"Sequence {seq} has fewer than 2 frames.")

        idx = random.randint(1, len(items) - 1)
        h, w = items[idx]["gt"].shape[:2]
        ps = min(patch_size, h, w)
        y0 = random.randint(0, h - ps)
        x0 = random.randint(0, w - ps)

        x_c = make_input_tensor(items, idx, device)[:, y0:y0+ps, x0:x0+ps]
        x_p = make_input_tensor(items, idx - 1, device)[:, y0:y0+ps, x0:x0+ps]
        g_c = to_torch_img(items[idx]["gt"], device)[:, y0:y0+ps, x0:x0+ps]
        g_p = to_torch_img(items[idx - 1]["gt"], device)[:, y0:y0+ps, x0:x0+ps]

        xs_cur.append(x_c)
        xs_prev.append(x_p)
        gt_cur.append(g_c)
        gt_prev.append(g_p)

    return (
        torch.stack(xs_cur, dim=0),
        torch.stack(xs_prev, dim=0),
        torch.stack(gt_cur, dim=0),
        torch.stack(gt_prev, dim=0),
    )


@torch.no_grad()
def run_model_on_sequence(model, items, device, tile=0):
    model.eval()
    outs = []

    for i in range(len(items)):
        x = make_input_tensor(items, i, device).unsqueeze(0)
        if tile and (x.shape[-1] > tile or x.shape[-2] > tile):
            # Simple non-overlap tiling. For this dataset max_side is usually modest, so this is rarely needed.
            _, _, h, w = x.shape
            out_full = torch.zeros((1, 3, h, w), device=device)
            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    ys = slice(y0, min(h, y0 + tile))
                    xs = slice(x0, min(w, x0 + tile))
                    out_patch, _ = model(x[:, :, ys, xs])
                    out_full[:, :, ys, xs] = out_patch
            out = out_full
        else:
            out, _ = model(x)

        img = out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        outs.append(np.clip(img, 0, 1))

    return outs


def eval_outputs(xs, items):
    gt = [it["gt"] for it in items]
    return {
        "num_frames": len(xs),
        "psnr": float(np.mean([psnr(x, y) for x, y in zip(xs, gt)])),
        "ssim": float(np.mean([ssim_rgb(x, y) for x, y in zip(xs, gt)])),
        "sharpness": float(np.mean([sharpness(x) for x in xs])),
        "tde": tde(xs, gt),
    }


def eval_fugr(items):
    return eval_outputs([it["fugr"] for it in items], items)


def train_one(data, args, temporal_weight, device, run_dir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = TinyTemporalRefiner(hidden=args.hidden, max_corr=args.max_corr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    train_log = []

    for step in range(1, args.steps + 1):
        model.train()
        x_c, x_p, g_c, g_p = sample_pair_batch(
            data, args.train_seqs, args.batch_size, args.patch_size, device
        )

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            out_c, corr_c = model(x_c)
            out_p, corr_p = model(x_p)

            loss_rec = 0.5 * charbonnier(out_c - g_c) + 0.5 * charbonnier(out_p - g_p)
            loss_temp = charbonnier((out_c - out_p) - (g_c - g_p))
            loss_reg = 0.5 * torch.mean(torch.abs(corr_c)) + 0.5 * torch.mean(torch.abs(corr_p))

            loss = loss_rec + temporal_weight * loss_temp + args.reg_weight * loss_reg

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_rec": float(loss_rec.detach().cpu()),
                "loss_temp": float(loss_temp.detach().cpu()),
                "loss_reg": float(loss_reg.detach().cpu()),
            }
            train_log.append(row)
            print(f"[tw={temporal_weight:.4f}] step {step}/{args.steps} "
                  f"loss={row['loss']:.6f} rec={row['loss_rec']:.6f} "
                  f"temp={row['loss_temp']:.6f} reg={row['loss_reg']:.6f}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "temporal_weight": temporal_weight,
        "args": vars(args),
    }, run_dir / "model.pt")

    with (run_dir / "train_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "loss_rec", "loss_temp", "loss_reg"])
        writer.writeheader()
        writer.writerows(train_log)

    return model


def evaluate_model(data, args, model, temporal_weight, device, out_dir):
    rows = []
    details = []
    split_defs = {
        "train": args.train_seqs,
        "test": args.test_seqs,
    }

    for split, seqs in split_defs.items():
        seq_rows = []
        for seq in seqs:
            items = data[seq]
            outs = run_model_on_sequence(model, items, device, tile=args.tile)
            rec = eval_outputs(outs, items)
            fugr = eval_fugr(items)
            row = {
                "temporal_weight": temporal_weight,
                "split": split,
                "sequence": seq,
                **rec,
                "fugr_psnr": fugr["psnr"],
                "fugr_ssim": fugr["ssim"],
                "fugr_sharpness": fugr["sharpness"],
                "fugr_tde": fugr["tde"],
                "delta_psnr_vs_fugr": rec["psnr"] - fugr["psnr"],
                "delta_ssim_vs_fugr": rec["ssim"] - fugr["ssim"],
                "delta_tde_vs_fugr": rec["tde"] - fugr["tde"],
            }
            details.append(row)
            seq_rows.append(row)

            if split == "test" and args.save_frames:
                for it, img in zip(items, outs):
                    save_rgb(out_dir / "frames" / f"tw_{temporal_weight:.4f}" / seq / f"{it['frame']}_a11.png", img)

        agg = {
            "temporal_weight": temporal_weight,
            "split": split,
            "num_sequences": len(seq_rows),
            "num_frames": int(sum(r["num_frames"] for r in seq_rows)),
            "psnr": float(np.mean([r["psnr"] for r in seq_rows])),
            "ssim": float(np.mean([r["ssim"] for r in seq_rows])),
            "sharpness": float(np.mean([r["sharpness"] for r in seq_rows])),
            "tde": float(np.mean([r["tde"] for r in seq_rows])),
            "fugr_psnr": float(np.mean([r["fugr_psnr"] for r in seq_rows])),
            "fugr_ssim": float(np.mean([r["fugr_ssim"] for r in seq_rows])),
            "fugr_sharpness": float(np.mean([r["fugr_sharpness"] for r in seq_rows])),
            "fugr_tde": float(np.mean([r["fugr_tde"] for r in seq_rows])),
        }
        agg["delta_psnr_vs_fugr"] = agg["psnr"] - agg["fugr_psnr"]
        agg["delta_ssim_vs_fugr"] = agg["ssim"] - agg["fugr_ssim"]
        agg["delta_tde_vs_fugr"] = agg["tde"] - agg["fugr_tde"]
        rows.append(agg)

    return rows, details


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_seqs", nargs="+", default=["000", "003", "006", "010", "011"])
    ap.add_argument("--test_seqs", nargs="+", default=["018", "020", "026", "028", "029"])
    ap.add_argument("--temporal_weights", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--reg_weight", type=float, default=0.05)
    ap.add_argument("--max_corr", type=float, default=0.04)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--save_frames", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    models_dir = out_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)

    data = collect(args.input_dir)
    print("Loaded sequences:", sorted(data.keys()), flush=True)

    missing = [s for s in args.train_seqs + args.test_seqs if s not in data]
    if missing:
        raise FileNotFoundError(f"Missing sequences: {missing}")

    all_summary = []
    all_details = []

    # Add FUGR-C reference rows.
    for split, seqs in {"train": args.train_seqs, "test": args.test_seqs}.items():
        seq_rows = []
        for seq in seqs:
            rec = eval_fugr(data[seq])
            seq_rows.append(rec)
        all_summary.append({
            "temporal_weight": -1.0,
            "split": split,
            "num_sequences": len(seqs),
            "num_frames": int(sum(r["num_frames"] for r in seq_rows)),
            "psnr": float(np.mean([r["psnr"] for r in seq_rows])),
            "ssim": float(np.mean([r["ssim"] for r in seq_rows])),
            "sharpness": float(np.mean([r["sharpness"] for r in seq_rows])),
            "tde": float(np.mean([r["tde"] for r in seq_rows])),
            "fugr_psnr": float(np.mean([r["psnr"] for r in seq_rows])),
            "fugr_ssim": float(np.mean([r["ssim"] for r in seq_rows])),
            "fugr_sharpness": float(np.mean([r["sharpness"] for r in seq_rows])),
            "fugr_tde": float(np.mean([r["tde"] for r in seq_rows])),
            "delta_psnr_vs_fugr": 0.0,
            "delta_ssim_vs_fugr": 0.0,
            "delta_tde_vs_fugr": 0.0,
        })

    for tw in args.temporal_weights:
        print(f"\n===== Training temporal_weight={tw:.4f} =====", flush=True)
        run_dir = models_dir / f"tw_{tw:.4f}"
        model = train_one(data, args, tw, device, run_dir)
        rows, details = evaluate_model(data, args, model, tw, device, out_dir)
        all_summary.extend(rows)
        all_details.extend(details)

        # Save partial after each run.
        write_csv(
            metrics_dir / "A11_summary_metrics.csv",
            all_summary,
            ["temporal_weight", "split", "num_sequences", "num_frames", "psnr", "ssim", "sharpness", "tde",
             "fugr_psnr", "fugr_ssim", "fugr_sharpness", "fugr_tde",
             "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_tde_vs_fugr"]
        )
        write_csv(
            metrics_dir / "A11_sequence_details.csv",
            all_details,
            ["temporal_weight", "split", "sequence", "num_frames", "psnr", "ssim", "sharpness", "tde",
             "fugr_psnr", "fugr_ssim", "fugr_sharpness", "fugr_tde",
             "delta_psnr_vs_fugr", "delta_ssim_vs_fugr", "delta_tde_vs_fugr"]
        )

    # Summary text.
    test_rows = [r for r in all_summary if r["split"] == "test"]
    best_test_psnr = sorted(test_rows, key=lambda r: (r["psnr"], -r["tde"]), reverse=True)[0]
    best_test_tde = sorted(test_rows, key=lambda r: (r["tde"], -r["psnr"]))[0]
    fugr_test = next(r for r in test_rows if r["temporal_weight"] < 0)

    txt = metrics_dir / "A11_temporal_refiner_summary.txt"
    with txt.open("w") as f:
        f.write("Direction A11: Train-time Temporal Fine-tuning with Tiny Temporal Refiner\n\n")
        f.write("FUGR-C test reference:\n")
        f.write(str(fugr_test) + "\n\n")
        f.write("Best test PSNR row:\n")
        f.write(str(best_test_psnr) + "\n\n")
        f.write("Best test TDE row:\n")
        f.write(str(best_test_tde) + "\n\n")
        f.write("Interpretation:\n")
        f.write(
            "A11 moves beyond fixed post-hoc rules by training a lightweight temporal refiner. "
            "If the trained refiner improves test PSNR or TDE, it can be reported as evidence that "
            "train-time temporal learning is more promising than rule-based post-processing. If it "
            "overfits calibration and fails on test, it reinforces the boundary-analysis conclusion.\n"
        )

    print(txt.read_text())
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
