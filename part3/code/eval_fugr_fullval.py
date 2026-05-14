#!/usr/bin/env python3
"""
Full-validation FUGR-VSR evaluation.

Run from Part3:
  python eval_fugr_fullval.py --out_dir /home/schung760/my_storage2_1T/AIAA3201_Part3_outputs/fullval_center7 --frame_mode center7
or for all 100 frames per sequence:
  python eval_fugr_fullval.py --out_dir /home/schung760/my_storage2_1T/AIAA3201_Part3_outputs/fullval_allframes --frame_mode all
"""

import argparse, csv, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch

PART2_DIR = Path(__file__).resolve().parents[1] / "Part2"
sys.path.insert(0, str(PART2_DIR))
from model_basicvsr import BasicVSR


def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def norm01(x, eps=1e-8):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


def gray(img):
    return cv2.cvtColor(np.clip(img*255,0,255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0


def psnr(x, y):
    if x.shape[:2] != y.shape[:2]:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
    mse = float(np.mean((x-y)**2))
    return 99.0 if mse < 1e-12 else float(20*np.log10(1.0/np.sqrt(mse)))


def ssim_rgb(x, y):
    if x.shape[:2] != y.shape[:2]:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
    c1, c2 = 0.01**2, 0.03**2
    vals = []
    for ch in range(3):
        a, b = x[..., ch].astype(np.float32), y[..., ch].astype(np.float32)
        ma, mb = cv2.GaussianBlur(a,(11,11),1.5), cv2.GaussianBlur(b,(11,11),1.5)
        va = cv2.GaussianBlur(a*a,(11,11),1.5) - ma*ma
        vb = cv2.GaussianBlur(b*b,(11,11),1.5) - mb*mb
        vab = cv2.GaussianBlur(a*b,(11,11),1.5) - ma*mb
        vals.append(float(np.mean(((2*ma*mb+c1)*(2*vab+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2)+1e-12))))
    return float(np.mean(vals))


def sharp(img):
    return float(np.var(cv2.Laplacian(gray(img), cv2.CV_32F)))


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma)


def texture(basic):
    g = gray(basic)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return norm01(cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy), (0,0), 1.2))


def disagreement(basic, gan):
    return cv2.GaussianBlur(np.mean(np.abs(basic-gan), axis=2), (0,0), 1.5)


def temporal_risk(prev_r, cur_r, next_r):
    if prev_r is None or next_r is None:
        return np.zeros(cur_r.shape[:2], dtype=np.float32)
    return cv2.GaussianBlur(np.mean(np.abs(cur_r - 0.5*(prev_r+next_r)), axis=2), (0,0), 1.2)


def alpha_map(basic, gan, prev_r, cur_r, next_r, amax, tau_dis, tau_temp, use_temp):
    dis = disagreement(basic, gan)
    rel_dis = np.exp(-dis / tau_dis)
    if use_temp:
        tr = temporal_risk(prev_r, cur_r, next_r)
        rel_temp = np.exp(-tr / tau_temp)
    else:
        tr = np.zeros_like(dis)
        rel_temp = np.ones_like(dis)
    a = amax * texture(basic) * rel_dis * rel_temp
    return np.clip(cv2.GaussianBlur(a, (0,0), 1.0), 0, amax), dis, tr


def rgb_blend(b, g, a):
    return np.clip((1-a[...,None])*b + a[...,None]*g, 0, 1)


def fugr(b, g, a, sigma, strength):
    detail = highpass(g, sigma) - highpass(b, sigma)
    return np.clip(b + strength*a[...,None]*detail, 0, 1)


def tde(outs, gts):
    if len(outs) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs((outs[i]-outs[i-1])-(gts[i]-gts[i-1]))) for i in range(1,len(outs))]))


def motion(gts):
    if len(gts) < 2:
        return 0.0
    return float(np.mean([np.mean(np.abs(gts[i]-gts[i-1])) for i in range(1,len(gts))]))


def load_model(ckpt_path, device, spynet_path):
    model = BasicVSR(spynet_path=spynet_path).to(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_lr_tensor(lr_dir, names):
    frames = []
    for n in names:
        img = read_rgb(lr_dir/n)
        frames.append(torch.from_numpy(img.transpose(2,0,1)).float())
    return torch.stack(frames, 0).unsqueeze(0)


@torch.no_grad()
def infer(model, lr_tensor, device, amp=False):
    x = lr_tensor.to(device)
    if device.type == "cuda" and amp:
        with torch.cuda.amp.autocast():
            y = model(x)
    else:
        y = model(x)
    y = torch.clamp(y[0].detach().cpu(), 0, 1).permute(0,2,3,1).numpy().astype(np.float32)
    del x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return [y[i] for i in range(y.shape[0])]


def eval_seq(seq, basic, gan, gts, args):
    residuals = [highpass(gan[i], args.hp_sigma)-highpass(basic[i], args.hp_sigma) for i in range(len(gts))]
    outs = {m: [] for m in ["BasicVSR","VSRGAN","RGB-Hybrid","FUGR-no-temporal","FUGR-temporal"]}
    alpha_means, dis_means, tr_means = [], [], []
    for i in range(len(gts)):
        b, g = basic[i], gan[i]
        rp = residuals[i-1] if i > 0 else None
        rc = residuals[i]
        rn = residuals[i+1] if i+1 < len(gts) else None
        a0, dis, tr0 = alpha_map(b,g,rp,rc,rn,args.max_alpha,args.tau_dis,args.tau_temp,False)
        at, dis, tr = alpha_map(b,g,rp,rc,rn,args.max_alpha,args.tau_dis,args.tau_temp,True)
        outs["BasicVSR"].append(b)
        outs["VSRGAN"].append(g)
        outs["RGB-Hybrid"].append(rgb_blend(b,g,at))
        outs["FUGR-no-temporal"].append(fugr(b,g,a0,args.hp_sigma,args.detail_strength))
        outs["FUGR-temporal"].append(fugr(b,g,at,args.hp_sigma,args.detail_strength))
        alpha_means.append(float(a0.mean())); dis_means.append(float(dis.mean())); tr_means.append(float(tr.mean()))
    rows = []
    mscore = motion(gts)
    for method, imgs in outs.items():
        rows.append({
            "sequence": seq, "method": method, "num_frames": len(gts),
            "psnr": float(np.mean([psnr(imgs[i], gts[i]) for i in range(len(gts))])),
            "ssim": float(np.mean([ssim_rgb(imgs[i], gts[i]) for i in range(len(gts))])),
            "laplacian_sharpness": float(np.mean([sharp(img) for img in imgs])),
            "tde": tde(imgs, gts), "motion": mscore,
            "alpha_mean": float(np.mean(alpha_means)),
            "disagreement_mean": float(np.mean(dis_means)),
            "temporal_risk_mean": float(np.mean(tr_means)),
        })
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def aggregate(rows):
    methods = sorted(set(r["method"] for r in rows))
    out = []
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        out.append({
            "method": m, "num_sequences": len(sub),
            "psnr": float(np.mean([r["psnr"] for r in sub])),
            "ssim": float(np.mean([r["ssim"] for r in sub])),
            "laplacian_sharpness": float(np.mean([r["laplacian_sharpness"] for r in sub])),
            "tde": float(np.mean([r["tde"] for r in sub])),
            "motion": float(np.mean([r["motion"] for r in sub])),
        })
    return sorted(out, key=lambda r: r["psnr"], reverse=True)


def motion_groups(rows):
    seq_motion = {}
    for r in rows:
        seq_motion[r["sequence"]] = r["motion"]
    seqs = sorted(seq_motion, key=lambda s: seq_motion[s])
    n = len(seqs)
    groups = {"low_motion": seqs[:n//3], "mid_motion": seqs[n//3:2*n//3], "high_motion": seqs[2*n//3:]}
    methods = sorted(set(r["method"] for r in rows))
    out = []
    for gname, gseqs in groups.items():
        for m in methods:
            sub = [r for r in rows if r["sequence"] in gseqs and r["method"] == m]
            if not sub: continue
            out.append({
                "group": gname, "method": m, "num_sequences": len(sub),
                "motion": float(np.mean([r["motion"] for r in sub])),
                "psnr": float(np.mean([r["psnr"] for r in sub])),
                "ssim": float(np.mean([r["ssim"] for r in sub])),
                "laplacian_sharpness": float(np.mean([r["laplacian_sharpness"] for r in sub])),
                "tde": float(np.mean([r["tde"] for r in sub])),
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_root", default="/home/schung760/shared_data/project1/val")
    p.add_argument("--basic_ckpt", default="/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_stage1.pth")
    p.add_argument("--gan_ckpt", default="/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_gan.pth")
    p.add_argument("--spynet_path", default="/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/spynet.pth")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_alpha", type=float, default=0.25)
    p.add_argument("--tau_dis", type=float, default=0.08)
    p.add_argument("--tau_temp", type=float, default=0.04)
    p.add_argument("--hp_sigma", type=float, default=1.6)
    p.add_argument("--detail_strength", type=float, default=1.2)
    p.add_argument("--frame_mode", choices=["center7","all"], default="center7")
    p.add_argument("--seq_limit", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val = Path(args.val_root); hr_root = val/"val_sharp"; lr_root = val/"val_sharp_bicubic"/"X4"
    seqs = sorted([d.name for d in hr_root.iterdir() if d.is_dir() and (lr_root/d.name).is_dir()])
    if args.seq_limit > 0: seqs = seqs[:args.seq_limit]
    print(f"Validation sequences: {len(seqs)}, frame_mode={args.frame_mode}, out={out_dir}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    basic_model = load_model(Path(args.basic_ckpt), device, args.spynet_path)
    gan_model = load_model(Path(args.gan_ckpt), device, args.spynet_path)

    all_rows = []
    for si, seq in enumerate(seqs):
        st = time.time()
        hr_dir, lr_dir = hr_root/seq, lr_root/seq
        names = sorted([p.name for p in hr_dir.glob("*.png") if (lr_dir/p.name).exists()])
        if args.frame_mode == "center7":
            names = names[:7]
        print(f"[{si+1}/{len(seqs)}] {seq}: {len(names)} frames", flush=True)
        gts = [read_rgb(hr_dir/n) for n in names]
        lr = load_lr_tensor(lr_dir, names)
        b = infer(basic_model, lr, device, args.amp); del lr
        lr = load_lr_tensor(lr_dir, names)
        g = infer(gan_model, lr, device, args.amp); del lr
        rows = eval_seq(seq, b, g, gts, args)
        all_rows.extend(rows)
        r = {x["method"]: x for x in rows}
        print(f"  Basic={r['BasicVSR']['psnr']:.4f}, GAN={r['VSRGAN']['psnr']:.4f}, "
              f"FUGR-noT={r['FUGR-no-temporal']['psnr']:.4f}, FUGR-T={r['FUGR-temporal']['psnr']:.4f}, "
              f"time={time.time()-st:.1f}s", flush=True)
        del gts, b, g
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seq_fields = ["sequence","method","num_frames","psnr","ssim","laplacian_sharpness","tde","motion","alpha_mean","disagreement_mean","temporal_risk_mean"]
    write_csv(out_dir/"fullval_per_sequence_metrics.csv", all_rows, seq_fields)
    summary = aggregate(all_rows)
    write_csv(out_dir/"fullval_summary_metrics.csv", summary, ["method","num_sequences","psnr","ssim","laplacian_sharpness","tde","motion"])
    mg = motion_groups(all_rows)
    write_csv(out_dir/"fullval_motion_group_metrics.csv", mg, ["group","method","num_sequences","motion","psnr","ssim","laplacian_sharpness","tde"])
    txt = out_dir/"fullval_summary.txt"
    with txt.open("w") as f:
        f.write("Full-validation FUGR-VSR evaluation\n")
        f.write(f"num_sequences: {len(seqs)}\nframe_mode: {args.frame_mode}\n")
        f.write(f"max_alpha: {args.max_alpha}\ntau_dis: {args.tau_dis}\ntau_temp: {args.tau_temp}\nhp_sigma: {args.hp_sigma}\ndetail_strength: {args.detail_strength}\n")
        f.write(f"total_time_sec: {time.time()-t0:.2f}\n\n")
        f.write("method,psnr,ssim,laplacian_sharpness,tde,motion,num_sequences\n")
        for r in summary:
            f.write(f"{r['method']},{r['psnr']:.6f},{r['ssim']:.6f},{r['laplacian_sharpness']:.8f},{r['tde']:.8f},{r['motion']:.8f},{r['num_sequences']}\n")
    print("\n===== Summary =====")
    print(txt.read_text())
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
