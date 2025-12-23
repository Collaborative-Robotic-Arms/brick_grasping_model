import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from resnet_grasp_neg.utils.targets import build_targets_from_center_labels


def _read_center_label_file(path):
    """
    Each line: cx cy ang_deg w h
    """
    out = []
    if not path or (not os.path.exists(path)):
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, ang, w, h = map(float, parts[:5])
            out.append((cx, cy, ang, w, h))
    return out


def _load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img  # H,W,3


def _load_depth(path):
    if path is None:
        return None
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        d = d.astype(np.float32)
    return d  # H,W float32


def _normalize_rgb(img):
    return img - img.mean()


def _normalize_depth(d):
    d = np.clip(d - float(d.mean()), -1.0, 1.0)
    return d


def _brightness_contrast_jitter(rgb, rng, p=0.6, brightness=0.20, contrast=0.25):
    if rng.random() > p:
        return rgb

    b = rng.uniform(-brightness, brightness)
    c = rng.uniform(1.0 - contrast, 1.0 + contrast)

    mean = rgb.mean(axis=(0, 1), keepdims=True)
    out = (rgb - mean) * c + mean + b
    out = np.clip(out, 0.0, 1.0)
    return out


def _random_blur(rgb, rng, p=0.15):
    if rng.random() > p:
        return rgb
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(rgb, (k, k), 0)


def _depth_gaussian_noise(depth, rng, p=0.5, sigma=0.03):
    if depth is None or (rng.random() > p):
        return depth
    noise = rng.normal(0.0, sigma, size=depth.shape).astype(np.float32)
    return depth + noise


class BrickGraspNegDataset(Dataset):
    """
    Structure expected:
      rgb/<base>.png
      depth/<base>.npy or <base>.png
      labels_pos/<base>cpos.txt
      labels_neg/<base>cneg.txt
    """
    def __init__(
        self,
        rgb_dir,
        depth_dir=None,
        labels_pos_dir=None,
        labels_neg_dir=None,
        output_size=160,
        include_rgb=True,
        include_depth=True,
        random_rotate=True,
        random_zoom=True,
        random_translate=True,
        seed=123,
        rgb_files=None,

        # ---- augmentation params ----
        rot_deg_jitter=15.0,
        zoom_min=0.7,
        zoom_max=1.3,
        translate_px=10,
        bcj_prob=0.6,
        bright=0.20,
        contrast=0.25,
        blur_prob=0.15,
        depth_noise_prob=0.5,
        depth_noise_sigma=0.03
    ):
        super().__init__()
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.labels_pos_dir = labels_pos_dir
        self.labels_neg_dir = labels_neg_dir
        self.output_size = int(output_size)
        self.include_rgb = bool(include_rgb)
        self.include_depth = bool(include_depth)

        self.random_rotate = bool(random_rotate)
        self.random_zoom = bool(random_zoom)
        self.random_translate = bool(random_translate)

        self.seed = int(seed)

        # aug hyperparams
        self.rot_deg_jitter = float(rot_deg_jitter)
        self.zoom_min = float(zoom_min)
        self.zoom_max = float(zoom_max)
        self.translate_px = int(translate_px)

        self.bcj_prob = float(bcj_prob)
        self.bright = float(bright)
        self.contrast = float(contrast)
        self.blur_prob = float(blur_prob)

        self.depth_noise_prob = float(depth_noise_prob)
        self.depth_noise_sigma = float(depth_noise_sigma)

        if (not self.include_rgb) and (not self.include_depth):
            raise ValueError("Choose include_rgb and/or include_depth")

        if rgb_files is not None:
            self.rgb_files = list(rgb_files)
        else:
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            files = []
            for e in exts:
                files += glob.glob(os.path.join(self.rgb_dir, e))
            files = sorted(files)
            if not files:
                raise RuntimeError(f"No RGB images found in {self.rgb_dir}")
            random.Random(self.seed).shuffle(files)
            self.rgb_files = files

    def __len__(self):
        return len(self.rgb_files)

    def _paths_for_index(self, idx):
        rgb_path = self.rgb_files[idx]
        base = os.path.splitext(os.path.basename(rgb_path))[0]

        depth_path = None
        if self.depth_dir is not None and self.include_depth:
            cand_npy = os.path.join(self.depth_dir, base + ".npy")
            cand_png = os.path.join(self.depth_dir, base + ".png")
            if os.path.exists(cand_npy):
                depth_path = cand_npy
            elif os.path.exists(cand_png):
                depth_path = cand_png
            else:
                raise FileNotFoundError(f"No depth file for {base} in {self.depth_dir}")

        pos_path = os.path.join(self.labels_pos_dir, base + "cpos.txt") if self.labels_pos_dir else None
        neg_path = os.path.join(self.labels_neg_dir, base + "cneg.txt") if self.labels_neg_dir else None
        return rgb_path, depth_path, pos_path, neg_path, base

    @staticmethod
    def _to_torch(arr):
        if arr.ndim == 2:
            return torch.from_numpy(arr[None, ...].astype(np.float32))
        else:
            return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))

    @staticmethod
    def _scale_labels(label_list, sx, sy):
        if not label_list:
            return label_list
        s_avg = 0.5 * (sx + sy)
        out = []
        for (cx, cy, ang, w, h) in label_list:
            out.append((cx * sx, cy * sy, ang, w * s_avg, h * s_avg))
        return out

    def __getitem__(self, idx):
        # IMPORTANT:
        # Make augmentation vary per call, but still controlled by worker seeds.
        # This draws a new seed from numpy's global RNG (seeded per worker in DataLoader).
        aug_seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
        rng = np.random.default_rng(aug_seed)

        rgb_path, depth_path, pos_path, neg_path, base = self._paths_for_index(idx)

        rgb = _load_rgb(rgb_path) if self.include_rgb else None
        depth = _load_depth(depth_path) if self.include_depth else None

        H0, W0 = (rgb.shape[0], rgb.shape[1]) if rgb is not None else (depth.shape[0], depth.shape[1])

        # read labels at original resolution
        pos_list = _read_center_label_file(pos_path)
        neg_list = _read_center_label_file(neg_path)

        # scale labels to output_size (if already 160x160 => sx=sy=1)
        sx = self.output_size / float(W0)
        sy = self.output_size / float(H0)
        pos_list = self._scale_labels(pos_list, sx, sy)
        neg_list = self._scale_labels(neg_list, sx, sy)

        # resize inputs to output_size
        if (H0 != self.output_size) or (W0 != self.output_size):
            if rgb is not None:
                rgb = cv2.resize(rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
            if depth is not None:
                depth = cv2.resize(depth, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)

        # ---- geometric augmentation: rotation + zoom + translation (single affine) ----
        angle_deg = 0.0
        zoom = 1.0
        tx = 0.0
        ty = 0.0

        if self.random_rotate:
            base_rot = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
            extra = float(rng.uniform(-self.rot_deg_jitter, self.rot_deg_jitter))
            angle_deg = base_rot + extra

        if self.random_zoom:
            zoom = float(rng.uniform(self.zoom_min, self.zoom_max))

        if self.random_translate and self.translate_px > 0:
            tx = float(rng.uniform(-self.translate_px, self.translate_px))
            ty = float(rng.uniform(-self.translate_px, self.translate_px))

        center = (self.output_size / 2.0, self.output_size / 2.0)  # (x,y)
        M = cv2.getRotationMatrix2D(center, angle_deg, zoom)
        M[:, 2] += np.array([tx, ty], dtype=np.float32)  # add translation

        if rgb is not None:
            rgb = cv2.warpAffine(rgb, M, (self.output_size, self.output_size),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
        if depth is not None:
            depth = cv2.warpAffine(depth, M, (self.output_size, self.output_size),
                                   flags=cv2.INTER_NEAREST, borderValue=0)

        # ---- appearance augmentation ----
        if rgb is not None:
            rgb = _brightness_contrast_jitter(rgb, rng, p=self.bcj_prob,
                                              brightness=self.bright, contrast=self.contrast)
            rgb = _random_blur(rgb, rng, p=self.blur_prob)

        if depth is not None:
            depth = _depth_gaussian_noise(depth, rng, p=self.depth_noise_prob, sigma=self.depth_noise_sigma)

        # ---- normalize ----
        if rgb is not None:
            rgb = _normalize_rgb(rgb)
        if depth is not None:
            depth = _normalize_depth(depth)

        # targets (NO WIDTH OUTPUT)
        pos, cos, sin, neg_mask, neg_cos, neg_sin = build_targets_from_center_labels(
            pos_list, neg_list, self.output_size, M_affine=M
        )

        # input tensor
        xs = []
        if self.include_depth:
            xs.append(self._to_torch(depth))  # 1,H,W
        if self.include_rgb:
            xs.append(self._to_torch(rgb))    # 3,H,W
        x = torch.cat(xs, dim=0)

        y = (
            self._to_torch(pos),
            self._to_torch(cos),
            self._to_torch(sin),
            self._to_torch(neg_mask),
            self._to_torch(neg_cos),
            self._to_torch(neg_sin),
        )

        meta = {"rgb_path": rgb_path, "base": base, "pos_path": pos_path, "idx": idx}
        return x, y, meta
