import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import lpips

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception as e:
    FrechetInceptionDistance = None
    print("Warning: FID unavailable:", e)


def read_img(path, device):
    img = Image.open(path).convert("RGB")
    return to_tensor(img).unsqueeze(0).to(device)


def common_names(pred_dir, gt_dir):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    pred_names = {p.name for p in pred_dir.glob("*.png")}
    gt_names = {p.name for p in gt_dir.glob("*.png")}
    return sorted(pred_names & gt_names)


def make_fid(device):
    if FrechetInceptionDistance is None:
        return None, True
    try:
        return FrechetInceptionDistance(feature=2048, normalize=True).to(device), True
    except TypeError:
        return FrechetInceptionDistance(feature=2048).to(device), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Format: Name=/path/to/method_root, where method_root/seq/*.png exists",
    )
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for item in args.methods:
        name, root = item.split("=", 1)
        root = Path(root)

        fid, fid_float = make_fid(device)
        lpips_vals = []
        tlpips_vals = []
        n_frames = 0

        for seq in args.seqs:
            pred_dir = root / seq
            gt_dir = Path(args.gt_root) / seq
            names = common_names(pred_dir, gt_dir)

            if not names:
                print(f"[Skip] {name} seq {seq}: no common PNG frames in {pred_dir}")
                continue

            prev_pred = None
            prev_gt = None

            for fname in names:
                pred = read_img(pred_dir / fname, device)
                gt = read_img(gt_dir / fname, device)

                with torch.no_grad():
                    # LPIPS expects inputs in [-1, 1].
                    lpips_vals.append(lpips_fn(pred * 2 - 1, gt * 2 - 1).item())

                    if fid is not None:
                        if fid_float:
                            fid.update(gt, real=True)
                            fid.update(pred, real=False)
                        else:
                            fid.update((gt * 255).byte(), real=True)
                            fid.update((pred * 255).byte(), real=False)

                    if prev_pred is not None:
                        # tLPIPS-style temporal metric:
                        # LPIPS between restored and GT temporal residuals.
                        dp = pred - prev_pred
                        dg = gt - prev_gt
                        tlpips_vals.append(lpips_fn(dp, dg).item())

                prev_pred = pred
                prev_gt = gt
                n_frames += 1

        lp = sum(lpips_vals) / max(len(lpips_vals), 1)
        tlp = sum(tlpips_vals) / max(len(tlpips_vals), 1)

        if fid is not None and n_frames > 1:
            try:
                fid_val = float(fid.compute().item())
            except Exception as e:
                print(f"FID failed for {name}:", e)
                fid_val = float("nan")
        else:
            fid_val = float("nan")

        rows.append(
            {
                "method": name,
                "num_frames": n_frames,
                "LPIPS": lp,
                "FID": fid_val,
                "tLPIPS_style": tlp,
            }
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "num_frames", "LPIPS", "FID", "tLPIPS_style"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
