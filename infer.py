import os
import argparse
import numpy as np
import torch
import cv2

from resnet_grasp_neg.models import ResNetUNetGraspNoWidth
from resnet_grasp_neg.utils.postprocess import post_process


def load_rgb(path, out_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img - img.mean()
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return img


def load_depth(path, out_size):
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = cv2.resize(d, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    d = np.clip(d - float(d.mean()), -1.0, 1.0)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", default=None)
    ap.add_argument("--out-size", type=int, default=160)
    ap.add_argument("--use-rgb", type=int, default=1)
    ap.add_argument("--use-depth", type=int, default=1)
    ap.add_argument("--q-thresh", type=float, default=0.5)
    ap.add_argument("--arch", default="segformer_b0",
                choices=["resnet_unet", "segformer_b0", "segformer_b2", "swin_tiny"])

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    include_rgb = bool(args.use_rgb)
    include_depth = bool(args.use_depth)

    xs = []
    if include_depth:
        if args.depth is None:
            raise ValueError("use-depth=1 but --depth is missing")
        d = load_depth(args.depth, args.out_size)
        xs.append(torch.from_numpy(d[None, ...]))
    if include_rgb:
        rgb = load_rgb(args.rgb, args.out_size)
        xs.append(torch.from_numpy(rgb.transpose(2, 0, 1)))

    x = torch.cat(xs, dim=0).float()[None, ...].to(device)

    in_ch = x.shape[1]
    model = ResNetUNetGraspNoWidth(in_channels=in_ch, pretrained=False).to(device)

    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        pred = model(x)
        q, ang = post_process(*pred)

    py, px = np.unravel_index(np.argmax(q), q.shape)
    theta = float(ang[py, px])
    score = float(q[py, px])

    print(f"BEST: x={px:.1f} y={py:.1f} theta(rad)={theta:.3f} q={score:.3f}")

    # gripper command 0|1 example
    close = 1 if score >= args.q_thresh else 0
    print(f"GRIPPER_CMD (0=open,1=close): {close}")


if __name__ == "__main__":
    main()
