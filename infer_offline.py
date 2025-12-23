#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import cv2
import torch

from resnet_grasp_neg.models import SwinGraspNoWidth, ResNetUNetGraspNoWidth


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img  # H,W,3


def load_depth(path):
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        d = d.astype(np.float32)
    return d  # H,W


def normalize_rgb(rgb):
    return rgb - rgb.mean()


def normalize_depth(d):
    d = np.clip(d - float(d.mean()), -1.0, 1.0)
    return d


def to_torch(arr):
    # arr H,W or H,W,C
    if arr.ndim == 2:
        return torch.from_numpy(arr[None, ...].astype(np.float32))
    else:
        return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))


def post_process(pos_logits, cos, sin):
    """
    pos_logits, cos, sin : torch tensors [B,1,H,W]
    Return:
      q: numpy [H,W] in [0,1]
      ang: numpy [H,W] in radians (-pi/2..pi/2)
    """
    pos = torch.sigmoid(pos_logits)
    ang = 0.5 * torch.atan2(sin, cos)
    q = pos
    q = q.squeeze().detach().cpu().numpy()
    ang = ang.squeeze().detach().cpu().numpy()
    return q, ang


def topk_points(q, k=5, min_dist=10):
    """
    Simple NMS-like top-k points on q map.
    """
    q2 = q.copy()
    pts = []
    H, W = q2.shape
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(q2), q2.shape)
        val = q2[y, x]
        if val <= 0:
            break
        pts.append((x, y, float(val)))
        # suppress neighborhood
        x0 = max(0, x - min_dist)
        x1 = min(W, x + min_dist + 1)
        y0 = max(0, y - min_dist)
        y1 = min(H, y + min_dist + 1)
        q2[y0:y1, x0:x1] = 0.0
    return pts


def grasp_corners(cx, cy, theta, length=50, width=20):
    """
    Build a rectangle (4 corners) centered at (cx,cy) with orientation theta.
    length: along grasp direction, width: perpendicular.
    """
    hl = length / 2.0
    hw = width / 2.0

    # direction unit vectors
    dx = np.cos(theta)
    dy = np.sin(theta)
    px = -np.sin(theta)
    py = np.cos(theta)

    p1 = (cx - hl*dx - hw*px, cy - hl*dy - hw*py)
    p2 = (cx + hl*dx - hw*px, cy + hl*dy - hw*py)
    p3 = (cx + hl*dx + hw*px, cy + hl*dy + hw*py)
    p4 = (cx - hl*dx + hw*px, cy - hl*dy + hw*py)
    return np.array([p1, p2, p3, p4], dtype=np.float32)


def draw_grasps(rgb, q, ang, topk=5, out_path=None, show=False):
    """
    rgb: float [0,1] RGB
    """
    vis = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).copy()

    pts = topk_points(q, k=topk, min_dist=12)
    for i, (x, y, score) in enumerate(pts):
        theta = float(ang[y, x])
        corners = grasp_corners(x, y, theta, length=50, width=20)
        corners_i = corners.astype(np.int32)

        # rectangle + center
        cv2.polylines(vis, [corners_i], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)

        cv2.putText(
            vis, f"{i}:{score:.2f}",
            (int(x)+5, int(y)-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA
        )

    # also save q heatmap overlay
    q_norm = (255 * (q - q.min()) / (q.max() - q.min() + 1e-6)).astype(np.uint8)
    heat = cv2.applyColorMap(q_norm, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted((vis).astype(np.uint8), 0.7, heat.astype(np.uint8), 0.3, 0)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if show:
        cv2.imshow("grasp_overlay", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    return overlay


def build_model(arch, in_ch, img_size, pretrained):
    arch = arch.lower()
    if arch == "swin_tiny":
        # IMPORTANT: your Swin class should accept img_size OR already handles it inside.
        return SwinGraspNoWidth(
            in_channels=in_ch,
            model_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            embed_dim=256,
            img_size=img_size
        )
    elif arch == "resnet_unet":
        return ResNetUNetGraspNoWidth(in_channels=in_ch, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported arch for offline inference: {arch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to BEST.pth or epoch_XXX.pth")
    ap.add_argument("--arch", default="swin_tiny", choices=["swin_tiny", "resnet_unet"])
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--depth-dir", default=None)
    ap.add_argument("--out-dir", default="infer_results")
    ap.add_argument("--input-size", type=int, default=160)
    ap.add_argument("--use-rgb", type=int, default=1)
    ap.add_argument("--use-depth", type=int, default=1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--pretrained", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    include_rgb = bool(args.use_rgb)
    include_depth = bool(args.use_depth)
    if (not include_rgb) and (not include_depth):
        raise ValueError("Choose --use-rgb 1 and/or --use-depth 1")

    in_ch = (3 if include_rgb else 0) + (1 if include_depth else 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, in_ch, args.input_size, bool(args.pretrained)).to(device)
    model.eval()

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    print(f"[OK] Loaded checkpoint: {args.ckpt}")

    # iterate over RGB images
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    rgb_files = []
    for e in exts:
        rgb_files += glob.glob(os.path.join(args.rgb_dir, e))
    rgb_files = sorted(rgb_files)
    if not rgb_files:
        raise RuntimeError(f"No RGB images found in {args.rgb_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for rgb_path in rgb_files:
            base = os.path.splitext(os.path.basename(rgb_path))[0]

            depth_path = None
            if include_depth:
                if args.depth_dir is None:
                    raise ValueError("You set --use-depth 1 but no --depth-dir")
                cand_npy = os.path.join(args.depth_dir, base + ".npy")
                cand_png = os.path.join(args.depth_dir, base + ".png")
                if os.path.exists(cand_npy):
                    depth_path = cand_npy
                elif os.path.exists(cand_png):
                    depth_path = cand_png
                else:
                    print(f"[SKIP] no depth for {base}")
                    continue

            rgb = load_rgb(rgb_path) if include_rgb else None
            depth = load_depth(depth_path) if include_depth else None

            # resize to input-size
            if include_rgb and (rgb.shape[0] != args.input_size or rgb.shape[1] != args.input_size):
                rgb = cv2.resize(rgb, (args.input_size, args.input_size), interpolation=cv2.INTER_LINEAR)
            if include_depth and (depth.shape[0] != args.input_size or depth.shape[1] != args.input_size):
                depth = cv2.resize(depth, (args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)

            # normalize like training
            xs = []
            if include_depth:
                depth_n = normalize_depth(depth)
                xs.append(to_torch(depth_n))
            if include_rgb:
                rgb_n = normalize_rgb(rgb)
                xs.append(to_torch(rgb_n))
            x = torch.cat(xs, dim=0)[None, ...].to(device)  # [1,C,H,W]

            pos_logits, cos, sin = model(x)
            q, ang = post_process(pos_logits, cos, sin)

            out_path = os.path.join(args.out_dir, f"{base}_overlay.png")
            draw_grasps(rgb if rgb is not None else np.repeat(depth[..., None], 3, axis=2),
                        q, ang, topk=args.topk, out_path=out_path, show=bool(args.show))

            print(f"[SAVE] {out_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
