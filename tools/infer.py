import argparse
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import TestDataset


def parse_args():
    parser = argparse.ArgumentParser(description="RaFBNet inference")
    parser.add_argument("--image_root", type=str, required=True, help="Directory of input RGB images.")
    parser.add_argument("--gt_root", type=str, default=None, help="Optional GT directory for resizing outputs.")
    parser.add_argument("--checkpoint", type=str, required=True, help="RaFBNet checkpoint path.")
    parser.add_argument("--save_root", type=str, default="outputs", help="Directory for predicted saliency maps.")
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pvt_checkpoint", type=str, default=str(ROOT / "weights" / "pvt_v2_b2.pth"))
    parser.add_argument("--radio_repo", type=str, required=True, help="Path to the RADIO repository.")
    parser.add_argument("--radio_checkpoint", type=str, required=True, help="Path to RADIO v2.5-B checkpoint.")
    parser.add_argument("--radio_gate_alpha", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    from models import RaFBNet

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = RaFBNet(
        pvt_checkpoint=args.pvt_checkpoint,
        radio_repo=args.radio_repo,
        radio_checkpoint=args.radio_checkpoint,
        radio_gate_alpha=args.radio_gate_alpha,
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    save_path = Path(args.save_root)
    save_path.mkdir(parents=True, exist_ok=True)
    test_loader = TestDataset(args.image_root, args.gt_root, args.testsize)

    time_sum = 0.0
    with torch.no_grad():
        for _ in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.to(device)
            start = time.time()
            pred, _ = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_sum += time.time() - start
            pred = F.interpolate(pred, size=gt.shape, mode="bilinear", align_corners=False)
            pred = pred.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            imageio.imsave(save_path / name, (pred * 255).astype(np.uint8))

    fps = test_loader.size / max(time_sum, 1e-8)
    print(f"Images: {test_loader.size}")
    print(f"Predictions: {save_path}")
    print(f"Average time: {time_sum / max(test_loader.size, 1):.5f}s")
    print(f"FPS: {fps:.5f}")


if __name__ == "__main__":
    main()
