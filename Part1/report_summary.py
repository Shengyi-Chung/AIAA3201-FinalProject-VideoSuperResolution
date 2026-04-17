from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a short markdown summary for Part1 results")
    parser.add_argument("--spatial-csv", type=Path, required=True)
    parser.add_argument("--temporal-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_spatial(path: Path) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            data[method] = {
                "psnr": float(row["psnr_y"]),
                "ssim": float(row["ssim_y"]),
                "fps": float(row["fps"]),
            }
    return data


def load_temporal_means(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    mean_row = None
    for row in rows:
        if row and row[0] == "MEAN":
            mean_row = row
            break

    if mean_row is None:
        raise RuntimeError("Cannot find MEAN row in temporal csv")

    return {
        "avg_psnr": float(mean_row[1]),
        "avg_ssim": float(mean_row[2]),
        "unsharp_psnr": float(mean_row[3]),
        "unsharp_ssim": float(mean_row[4]),
    }


def main() -> None:
    args = parse_args()

    spatial = load_spatial(args.spatial_csv)
    temporal = load_temporal_means(args.temporal_csv)

    bicubic = spatial.get("bicubic")
    srcnn = spatial.get("srcnn")

    lines = []
    lines.append("# Part1 Result Summary")
    lines.append("")
    lines.append("## Spatial Baseline")
    lines.append("")
    lines.append("| Method | PSNR(Y) | SSIM(Y) | FPS |")
    lines.append("|---|---:|---:|---:|")
    for method, vals in spatial.items():
        lines.append(f"| {method} | {vals['psnr']:.4f} | {vals['ssim']:.4f} | {vals['fps']:.2f} |")

    lines.append("")
    lines.append("## Temporal Baseline (Mean)")
    lines.append("")
    lines.append(f"- temporal_avg: PSNR={temporal['avg_psnr']:.4f}, SSIM={temporal['avg_ssim']:.4f}")
    lines.append(f"- temporal_avg_unsharp: PSNR={temporal['unsharp_psnr']:.4f}, SSIM={temporal['unsharp_ssim']:.4f}")

    if bicubic is not None and srcnn is not None:
        d_psnr = srcnn["psnr"] - bicubic["psnr"]
        d_ssim = srcnn["ssim"] - bicubic["ssim"]
        lines.append("")
        lines.append("## Quick Interpretation")
        lines.append("")
        lines.append(f"- SRCNN vs Bicubic: dPSNR={d_psnr:+.4f}, dSSIM={d_ssim:+.4f}")
        lines.append("- Temporal averaging improves denoising but may introduce motion blur/ghosting.")
        lines.append("- Unsharp masking can recover edge contrast while risking slight ringing/noise amplification.")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to: {args.out}")


if __name__ == "__main__":
    main()
