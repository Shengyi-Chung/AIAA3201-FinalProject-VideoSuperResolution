import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from model_basicvsr import BasicVSR


def run_one_sequence(model, device, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".png"])
    if not all_imgs:
        print(f"[Skip] no png frames: {input_dir}")
        return

    lr_frames = []
    for p in all_imgs:
        img = Image.open(p).convert("RGB")
        lr_frames.append(ToTensor()(img))

    with torch.no_grad():
        x = torch.stack(lr_frames).unsqueeze(0).to(device)
        y = model(x)
        for i, p in enumerate(all_imgs):
            save_image(y[0, i], output_dir / p.name)

    print(f"[OK] {input_dir.name}: {len(all_imgs)} frames -> {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--spynet", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicVSR(spynet_path=args.spynet).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    for seq in args.seqs:
        run_one_sequence(
            model=model,
            device=device,
            input_dir=Path(args.lr_root) / seq,
            output_dir=Path(args.out_root) / seq,
        )


if __name__ == "__main__":
    main()
