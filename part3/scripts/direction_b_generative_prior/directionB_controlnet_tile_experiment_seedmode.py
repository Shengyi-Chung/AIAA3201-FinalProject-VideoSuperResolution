#!/usr/bin/env python3
"""
Direction B: ControlNet-Tile generative-prior VSR experiment.

It compares:
- BasicVSR
- VSRGAN
- FUGR-C
- ControlNet-Basic
- ControlNet-FUGR

Outputs:
- frames/
- panels/
- metrics/directionB_frame_metrics.csv
- metrics/directionB_summary_metrics.csv
- metrics/directionB_tde_metrics.csv
- metrics/directionB_summary.txt
"""

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PART2_DIR = Path(__file__).resolve().parents[1] / "Part2"
sys.path.insert(0, str(PART2_DIR))
from model_basicvsr import BasicVSR


def read_rgb(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_rgb(p: Path, img: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(p), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def np_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(img * 255.0, 0, 255).round().astype(np.uint8))


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0


def resize_to_multiple_of_8(img: np.ndarray, max_side: int = 768):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    new_h = max(64, int(round(h * scale / 8) * 8))
    new_w = max(64, int(round(w * scale / 8) * 8))
    if (new_h, new_w) == (h, w):
        return img, (h, w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC), (h, w)


def restore_size(img: np.ndarray, target_hw):
    h, w = target_hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def gray(img):
    return cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def norm01(x, eps=1e-8):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


def colorize(x):
    u8 = np.clip(norm01(x) * 255, 0, 255).astype(np.uint8)
    c = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def err_map(x, gt):
    return colorize(np.mean(np.abs(x - gt), axis=2))


def resize_to_h(img, target_h=240):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)


def make_panel(path: Path, imgs, titles, target_h=240):
    imgs = [resize_to_h(i, target_h) for i in imgs]
    title_h = 34
    canvas = np.ones((target_h + title_h, sum(i.shape[1] for i in imgs), 3), dtype=np.uint8) * 255
    x = 0
    for img, title in zip(imgs, titles):
        w = img.shape[1]
        canvas[title_h:, x:x+w] = np.clip(img * 255, 0, 255).round().astype(np.uint8)
        cv2.putText(canvas, title, (x + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        x += w
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def psnr(x, y):
    if x.shape[:2] != y.shape[:2]:
        x = cv2.resize(x, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_CUBIC)
    mse = float(np.mean((x - y) ** 2))
    return 99.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def ssim_rgb(x, y):
    if x.shape[:2] != y.shape[:2]:
        x = cv2.resize(x, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_CUBIC)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        a = x[..., ch].astype(np.float32)
        b = y[..., ch].astype(np.float32)
        ma = cv2.GaussianBlur(a, (11, 11), 1.5)
        mb = cv2.GaussianBlur(b, (11, 11), 1.5)
        va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - ma * ma
        vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mb * mb
        vab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - ma * mb
        vals.append(float(np.mean(((2 * ma * mb + c1) * (2 * vab + c2)) /
                                  ((ma * ma + mb * mb + c1) * (va + vb + c2) + 1e-12))))
    return float(np.mean(vals))


def sharpness(img):
    return float(np.var(cv2.Laplacian(gray(img), cv2.CV_32F)))


def tde(outputs, gts):
    if len(outputs) < 2:
        return 0.0
    return float(np.mean([
        np.mean(np.abs((outputs[i] - outputs[i-1]) - (gts[i] - gts[i-1])))
        for i in range(1, len(outputs))
    ]))


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def texture_map(basic):
    g = gray(basic)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return norm01(cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2))


def disagreement_map(basic, gan):
    return cv2.GaussianBlur(np.mean(np.abs(basic - gan), axis=2), (0, 0), 1.5)


def fugr_frame(basic, gan, max_alpha=0.25, tau_dis=0.08, hp_sigma=1.6, detail_strength=1.2):
    tex = texture_map(basic)
    dis = disagreement_map(basic, gan)
    alpha = np.clip(cv2.GaussianBlur(max_alpha * tex * np.exp(-dis / tau_dis), (0, 0), 1.0), 0, max_alpha)
    detail = highpass(gan, hp_sigma) - highpass(basic, hp_sigma)
    return np.clip(basic + detail_strength * alpha[..., None] * detail, 0, 1), alpha


def load_model(ckpt_path, device, spynet_path):
    model = BasicVSR(spynet_path=spynet_path).to(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_lr(lr_dir, names):
    arr = []
    for n in names:
        img = read_rgb(lr_dir / n)
        arr.append(torch.from_numpy(img.transpose(2, 0, 1)).float())
    return torch.stack(arr, dim=0).unsqueeze(0)


@torch.no_grad()
def infer(model, lr_tensor, device, amp=False):
    x = lr_tensor.to(device)
    if device.type == "cuda" and amp:
        with torch.cuda.amp.autocast():
            y = model(x)
    else:
        y = model(x)
    y = torch.clamp(y[0].detach().cpu(), 0, 1).permute(0, 2, 3, 1).numpy().astype(np.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [y[i] for i in range(y.shape[0])]


def get_indices(names, args):
    if args.mode == "selected":
        return [min(max(i, 0), len(names) - 1) for i in args.frame_indices]
    start = max(0, args.clip_center - args.clip_len // 2)
    end = min(len(names), start + args.clip_len)
    start = max(0, end - args.clip_len)
    return list(range(start, end))


def load_controlnet_pipe(args):
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model, torch_dtype=torch.float16, cache_dir=os.environ.get("HF_HOME")
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=os.environ.get("HF_HOME"),
    )
    pipe.enable_attention_slicing()
    return pipe.to("cuda")


@torch.no_grad()
def enhance(pipe, img, args, seed):
    img8, orig_hw = resize_to_multiple_of_8(img, args.max_side)
    pil = np_to_pil(img8)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=pil,
        control_image=pil,
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.control_scale,
        generator=gen,
    ).images[0]
    return restore_size(pil_to_np(out), orig_hw)


def metric_row(seq, frame, method, img, gt):
    return {
        "sequence": seq, "frame": frame, "method": method,
        "psnr": psnr(img, gt), "ssim": ssim_rgb(img, gt), "laplacian_sharpness": sharpness(img)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_root", default="/home/schung760/shared_data/project1/val")
    ap.add_argument("--basic_ckpt", default="/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_stage1.pth")
    ap.add_argument("--gan_ckpt", default="/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/basicvsr_gan.pth")
    ap.add_argument("--spynet_path", default="/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part2/weights/spynet.pth")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--controlnet_model", default="lllyasviel/control_v11f1e_sd15_tile")
    ap.add_argument("--prompt", default="high quality realistic video frame, sharp natural details, clean edges, faithful colors")
    ap.add_argument("--negative_prompt", default="low quality, blurry, noisy, artifacts, distorted, oversharpened, hallucinated details, text, watermark")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.25)
    ap.add_argument("--guidance_scale", type=float, default=7.0)
    ap.add_argument("--control_scale", type=float, default=0.75)
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seed_mode", choices=["incremental", "fixed"], default="incremental")
    ap.add_argument("--seqs", nargs="+", default=["028", "003", "020"])
    ap.add_argument("--mode", choices=["selected", "clip"], default="selected")
    ap.add_argument("--frame_indices", nargs="+", type=int, default=[25, 50, 75])
    ap.add_argument("--clip_center", type=int, default=50)
    ap.add_argument("--clip_len", type=int, default=7)
    ap.add_argument("--max_alpha", type=float, default=0.25)
    ap.add_argument("--tau_dis", type=float, default=0.08)
    ap.add_argument("--hp_sigma", type=float, default=1.6)
    ap.add_argument("--detail_strength", type=float, default=1.2)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    panels_dir = out_dir / "panels"
    metrics_dir = out_dir / "metrics"
    for d in [frames_dir, panels_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    val_root = Path(args.val_root)
    hr_root = val_root / "val_sharp"
    lr_root = val_root / "val_sharp_bicubic" / "X4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    print("Preparing BasicVSR / VSRGAN / FUGR-C outputs...", flush=True)
    basic_model = load_model(Path(args.basic_ckpt), device, args.spynet_path)
    gan_model = load_model(Path(args.gan_ckpt), device, args.spynet_path)

    data = {}
    metrics = []
    for seq in args.seqs:
        hr_dir, lr_dir = hr_root / seq, lr_root / seq
        names = sorted([p.name for p in hr_dir.glob("*.png") if (lr_dir / p.name).exists()])
        idxs = get_indices(names, args)
        selected = [names[i] for i in idxs]
        print(f"Sequence {seq}: {selected}", flush=True)

        lr = load_lr(lr_dir, selected)
        basics = infer(basic_model, lr, device, args.amp)
        del lr
        lr = load_lr(lr_dir, selected)
        gans = infer(gan_model, lr, device, args.amp)
        del lr

        gts = [read_rgb(hr_dir / n) for n in selected]
        fugrs, alphas = [], []
        for b, g in zip(basics, gans):
            f, a = fugr_frame(b, g, args.max_alpha, args.tau_dis, args.hp_sigma, args.detail_strength)
            fugrs.append(f)
            alphas.append(a)

        data[seq] = {"names": selected, "gt": gts, "basic": basics, "gan": gans, "fugr": fugrs, "alpha": alphas}

        for n, b, g, f, gt in zip(selected, basics, gans, fugrs, gts):
            metrics += [metric_row(seq, n, "BasicVSR", b, gt), metric_row(seq, n, "VSRGAN", g, gt), metric_row(seq, n, "FUGR-C", f, gt)]

    del basic_model, gan_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading ControlNet-Tile Img2Img pipeline...", flush=True)
    pipe = load_controlnet_pipe(args)
    print("Pipeline loaded.", flush=True)

    for seq, d in data.items():
        cb_list, cf_list = [], []
        for i, n in enumerate(d["names"]):
            print(f"ControlNet enhancing seq={seq}, frame={n} ({i+1}/{len(d['names'])})", flush=True)
            b, f, gt = d["basic"][i], d["fugr"][i], d["gt"][i]
            seed_b = args.seed if args.seed_mode == "fixed" else args.seed + i
            seed_f = args.seed + 1000 if args.seed_mode == "fixed" else args.seed + 1000 + i
            cb = enhance(pipe, b, args, seed_b)
            cf = enhance(pipe, f, args, seed_f)
            cb_list.append(cb)
            cf_list.append(cf)

            stem = Path(n).stem
            save_rgb(frames_dir / seq / f"{stem}_basic.png", b)
            save_rgb(frames_dir / seq / f"{stem}_fugr.png", f)
            save_rgb(frames_dir / seq / f"{stem}_controlnet_basic.png", cb)
            save_rgb(frames_dir / seq / f"{stem}_controlnet_fugr.png", cf)
            save_rgb(frames_dir / seq / f"{stem}_gt.png", gt)
            save_rgb(frames_dir / seq / f"{stem}_alpha.png", colorize(d["alpha"][i]))

            metrics += [metric_row(seq, n, "ControlNet-Basic", cb, gt), metric_row(seq, n, "ControlNet-FUGR", cf, gt)]

            make_panel(
                panels_dir / f"panel_seq{seq}_{stem}.png",
                [b, f, cb, cf, gt, err_map(f, gt), err_map(cf, gt), colorize(d["alpha"][i])],
                ["Basic", "FUGR-C", "CN-Basic", "CN-FUGR", "GT", "FUGR Err", "CN-FUGR Err", "Alpha"],
            )
        d["controlnet_basic"] = cb_list
        d["controlnet_fugr"] = cf_list

    frame_csv = metrics_dir / "directionB_frame_metrics.csv"
    with frame_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "frame", "method", "psnr", "ssim", "laplacian_sharpness"])
        w.writeheader()
        w.writerows(metrics)

    methods = sorted(set(r["method"] for r in metrics))
    summary_rows = []
    for m in methods:
        sub = [r for r in metrics if r["method"] == m]
        summary_rows.append({
            "method": m, "num_frames": len(sub),
            "psnr": float(np.mean([r["psnr"] for r in sub])),
            "ssim": float(np.mean([r["ssim"] for r in sub])),
            "laplacian_sharpness": float(np.mean([r["laplacian_sharpness"] for r in sub])),
        })

    summary_csv = metrics_dir / "directionB_summary_metrics.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "num_frames", "psnr", "ssim", "laplacian_sharpness"])
        w.writeheader()
        w.writerows(sorted(summary_rows, key=lambda x: x["psnr"], reverse=True))

    tde_rows = []
    for seq, d in data.items():
        if len(d["names"]) >= 2:
            for method, outs in [
                ("BasicVSR", d["basic"]),
                ("VSRGAN", d["gan"]),
                ("FUGR-C", d["fugr"]),
                ("ControlNet-Basic", d["controlnet_basic"]),
                ("ControlNet-FUGR", d["controlnet_fugr"]),
            ]:
                tde_rows.append({"sequence": seq, "method": method, "tde": tde(outs, d["gt"]), "num_frames": len(d["names"])})

    tde_csv = metrics_dir / "directionB_tde_metrics.csv"
    with tde_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "method", "tde", "num_frames"])
        w.writeheader()
        w.writerows(tde_rows)

    txt = metrics_dir / "directionB_summary.txt"
    with txt.open("w") as f:
        f.write("Direction B: ControlNet-Tile generative-prior experiment\n")
        f.write(f"mode: {args.mode}\nseqs: {args.seqs}\nsteps: {args.steps}\nstrength: {args.strength}\ncontrol_scale: {args.control_scale}\nguidance_scale: {args.guidance_scale}\n\n")
        f.write("Frame-level summary:\n")
        for r in sorted(summary_rows, key=lambda x: x["psnr"], reverse=True):
            f.write(f"{r['method']},{r['num_frames']},{r['psnr']:.6f},{r['ssim']:.6f},{r['laplacian_sharpness']:.8f}\n")
        if tde_rows:
            f.write("\nTDE summary by sequence:\n")
            for r in tde_rows:
                f.write(f"{r['sequence']},{r['method']},{r['tde']:.8f},{r['num_frames']}\n")

    print("\n===== Direction B summary =====")
    print(txt.read_text())
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
