#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import cv2
import torch

from ultralytics import YOLO  # YOLOv8/YOLO11 API
from resnet_grasp_neg.models import SwinGraspNoWidth, ResNetUNetGraspNoWidth


# ----------------- IO + normalize (same as training) -----------------
def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img

def load_depth(path):
    if path is None:
        return None
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return d.astype(np.float32)

def normalize_rgb(rgb):
    return rgb - rgb.mean()

def normalize_depth(d):
    return np.clip(d - float(d.mean()), -1.0, 1.0)

def to_torch(arr):
    if arr.ndim == 2:
        return torch.from_numpy(arr[None, ...].astype(np.float32))
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))

def post_process(pos_logits, cos, sin):
    pos = torch.sigmoid(pos_logits)
    ang = 0.5 * torch.atan2(sin, cos)
    q = pos.squeeze().detach().cpu().numpy()
    ang = ang.squeeze().detach().cpu().numpy()
    return q, ang


# ----------------- mask utils -----------------
def union_mask_from_yolo(result, target_class=None, conf_thresh=0.25, out_hw=None):
    """
    result: ultralytics Results (single image)
    returns binary mask uint8 {0,1} shape (H,W) or None
    """
    if result.masks is None or result.masks.data is None:
        return None

    m = result.masks.data  # torch (N, Hm, Wm)
    if m.numel() == 0:
        return None

    # optional filter by class/conf
    if result.boxes is not None and result.boxes.cls is not None and result.boxes.conf is not None:
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        conf = result.boxes.conf.detach().cpu().numpy()
        keep = conf >= conf_thresh
        if target_class is not None:
            keep = keep & (cls == int(target_class))
        if keep.any():
            m = m[torch.from_numpy(keep).to(m.device)]

    if m.numel() == 0:
        return None

    mask = (m.sum(dim=0) > 0).detach().cpu().numpy().astype(np.uint8)  # (Hm, Wm)

    if out_hw is not None and (mask.shape[0] != out_hw[0] or mask.shape[1] != out_hw[1]):
        mask = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)

    return mask

def erode_mask(mask, k=9, it=1):
    if mask is None:
        return None
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=it)

def pick_top1_inside_mask(q, mask_eroded):
    """
    q: float HxW
    mask_eroded: uint8 HxW {0,1}
    returns (x,y,score) or None
    """
    if mask_eroded is None or mask_eroded.sum() == 0:
        # fallback: global argmax
        y, x = np.unravel_index(np.argmax(q), q.shape)
        return int(x), int(y), float(q[y, x])

    q_use = q * (mask_eroded.astype(q.dtype))
    if q_use.max() <= 0:
        y, x = np.unravel_index(np.argmax(q), q.shape)
        return int(x), int(y), float(q[y, x])

    y, x = np.unravel_index(np.argmax(q_use), q_use.shape)
    return int(x), int(y), float(q_use[y, x])


# ----------------- draw grasp -----------------
def grasp_corners(cx, cy, theta, length=50, width=20):
    hl = length / 2.0
    hw = width / 2.0
    dx, dy = np.cos(theta), np.sin(theta)
    px, py = -np.sin(theta), np.cos(theta)
    p1 = (cx - hl*dx - hw*px, cy - hl*dy - hw*py)
    p2 = (cx + hl*dx - hw*px, cy + hl*dy - hw*py)
    p3 = (cx + hl*dx + hw*px, cy + hl*dy + hw*py)
    p4 = (cx - hl*dx + hw*px, cy - hl*dy + hw*py)
    return np.array([p1, p2, p3, p4], dtype=np.float32)

def draw_overlay(rgb, mask, x, y, score, theta, q=None, out_path=None):
    vis = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).copy()

    # draw mask contour
    if mask is not None and mask.sum() > 0:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 0), 2)

    corners = grasp_corners(x, y, theta, length=50, width=20).astype(np.int32)
    cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
    cv2.circle(vis, (x, y), 3, (255, 0, 0), -1)
    cv2.putText(vis, f"q={score:.2f}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    # optional heat overlay
    if q is not None:
        qn = (255 * (q - q.min()) / (q.max() - q.min() + 1e-6)).astype(np.uint8)
        heat = cv2.applyColorMap(qn, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        vis = cv2.addWeighted(vis, 0.7, heat, 0.3, 0)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return vis


# ----------------- build grasp model -----------------
def build_grasp_model(arch, in_ch, img_size, pretrained):
    arch = arch.lower()
    if arch == "swin_tiny":
        return SwinGraspNoWidth(
            in_channels=in_ch,
            model_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            embed_dim=256,
            img_size=img_size
        )
    if arch == "resnet_unet":
        return ResNetUNetGraspNoWidth(in_channels=in_ch, pretrained=pretrained)
    raise ValueError(f"Unknown arch: {arch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-weights", required=True, help="path to YOLO-seg weights .pt")
    ap.add_argument("--grasp-ckpt", required=True, help="path to grasp checkpoint BEST.pth")
    ap.add_argument("--grasp-arch", default="swin_tiny", choices=["swin_tiny", "resnet_unet"])
    ap.add_argument("--grasp-pretrained", type=int, default=0)

    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--depth-dir", default=None)
    ap.add_argument("--use-rgb", type=int, default=1)
    ap.add_argument("--use-depth", type=int, default=1)
    ap.add_argument("--input-size", type=int, default=160)

    ap.add_argument("--out-dir", default="infer_yolo_mask_results")

    ap.add_argument("--yolo-imgsz", type=int, default=160)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-iou", type=float, default=0.7)
    ap.add_argument("--yolo-class", type=int, default=None, help="optional class id to keep")
    ap.add_argument("--retina-masks", type=int, default=1)

    ap.add_argument("--erode-k", type=int, default=9)
    ap.add_argument("--erode-it", type=int, default=1)

    args = ap.parse_args()

    include_rgb = bool(args.use_rgb)
    include_depth = bool(args.use_depth)
    if (not include_rgb) and (not include_depth):
        raise ValueError("Choose --use-rgb 1 and/or --use-depth 1")

    in_ch = (3 if include_rgb else 0) + (1 if include_depth else 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # YOLO seg (offline: weights local)
    yolo = YOLO(args.yolo_weights)

    # Grasp model
    grasp_model = build_grasp_model(args.grasp_arch, in_ch, args.input_size, bool(args.grasp_pretrained)).to(device)
    grasp_model.eval()
    ck = torch.load(args.grasp_ckpt, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    grasp_model.load_state_dict(sd, strict=True)

    # files
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    rgb_files = []
    for e in exts:
        rgb_files += glob.glob(os.path.join(args.rgb_dir, e))
    rgb_files = sorted(rgb_files)
    if not rgb_files:
        raise RuntimeError(f"No images in {args.rgb_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for rgb_path in rgb_files:
            base = os.path.splitext(os.path.basename(rgb_path))[0]

            depth_path = None
            if include_depth:
                cand_npy = os.path.join(args.depth_dir, base + ".npy")
                cand_png = os.path.join(args.depth_dir, base + ".png")
                depth_path = cand_npy if os.path.exists(cand_npy) else cand_png if os.path.exists(cand_png) else None

            rgb = load_rgb(rgb_path) if include_rgb else None
            depth = load_depth(depth_path) if include_depth else None

            # resize to grasp input
            if include_rgb and (rgb.shape[0] != args.input_size or rgb.shape[1] != args.input_size):
                rgb = cv2.resize(rgb, (args.input_size, args.input_size), interpolation=cv2.INTER_LINEAR)
            if include_depth and depth is not None and (depth.shape[0] != args.input_size or depth.shape[1] != args.input_size):
                depth = cv2.resize(depth, (args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)

            # ---- YOLO segmentation on RGB (uses same resized rgb) ----
            # retina_masks=True => masks.data match image size (important) :contentReference[oaicite:2]{index=2}
            pred = yolo.predict(
                source=cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                imgsz=args.yolo_imgsz,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                retina_masks=bool(args.retina_masks),
                verbose=False
            )
            r = pred[0]
            mask = union_mask_from_yolo(
                r,
                target_class=args.yolo_class,
                conf_thresh=args.yolo_conf,
                out_hw=(args.input_size, args.input_size)
            )
            mask_er = erode_mask(mask, k=args.erode_k, it=args.erode_it)

            # ---- grasp forward ----
            xs = []
            if include_depth:
                if depth is None:
                    print(f"[SKIP] no depth for {base}")
                    continue
                xs.append(to_torch(normalize_depth(depth)))
            if include_rgb:
                xs.append(to_torch(normalize_rgb(rgb)))

            x = torch.cat(xs, dim=0)[None, ...].to(device)  # [1,C,H,W]
            pos_logits, cos, sin = grasp_model(x)
            q, ang = post_process(pos_logits, cos, sin)

            # ---- pick Top-1 INSIDE eroded mask ----
            x0, y0, score = pick_top1_inside_mask(q, mask_er)
            theta = float(ang[y0, x0])
