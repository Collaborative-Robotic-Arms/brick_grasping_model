#!/usr/bin/env python3
import os
import time
import argparse
import random
import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from resnet_grasp_neg.datasets import BrickGraspNegDataset
from resnet_grasp_neg.models import ResNetUNetGraspNoWidth, SegFormerGraspNoWidth, SwinGraspNoWidth
from resnet_grasp_neg.losses import GraspLossWithNegativesNoWidth
from resnet_grasp_neg.utils.postprocess import post_process


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # Ensures each dataloader worker has a different, reproducible seed
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_backbone_trainable(model, trainable: bool):
    if hasattr(model, "backbone"):  # SegFormer / Swin (timm wrapper)
        for p in model.backbone.parameters():
            p.requires_grad = trainable
        return
    if hasattr(model, "enc"):       # ResNet UNet
        for p in model.enc.parameters():
            p.requires_grad = trainable
        return
    print("[WARN] Could not find backbone to freeze/unfreeze on this model.")


def center_angle_match(q_img, ang_img, pos_labels, dist_thresh=20.0, angle_thresh_deg=30.0):
    if len(pos_labels) == 0:
        return False

    py, px = np.unravel_index(np.argmax(q_img), q_img.shape)
    ptheta = float(ang_img[py, px])

    for (cx, cy, ang_deg, w, h) in pos_labels:
        gcx, gcy = float(cx), float(cy)
        gtheta = np.deg2rad(float(ang_deg))

        dist = np.sqrt((px - gcx) ** 2 + (py - gcy) ** 2)

        ang_diff = abs(ptheta - gtheta)
        ang_diff = min(ang_diff, np.pi - ang_diff)

        if dist <= dist_thresh and ang_diff <= np.deg2rad(angle_thresh_deg):
            return True

    return False


def read_pos_file(pos_path):
    out = []
    if not pos_path or (not os.path.exists(pos_path)):
        return out
    with open(pos_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, a, ww, hh = map(float, parts[:5])
                out.append((cx, cy, a, ww, hh))
    return out


def _make_dataset_kwargs(**kwargs):
    sig = inspect.signature(BrickGraspNegDataset.__init__)
    allowed = set(sig.parameters.keys())
    out = {}
    for k, v in kwargs.items():
        if k in allowed:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--depth-dir", default=None)
    ap.add_argument("--labels-pos", required=True)
    ap.add_argument("--labels-neg", required=True)

    ap.add_argument("--output-size", type=int, default=160)
    ap.add_argument("--use-rgb", type=int, default=1)
    ap.add_argument("--use-depth", type=int, default=1)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--w-neg-q", type=float, default=0.5)
    ap.add_argument("--w-neg-ang", type=float, default=0.25)
    ap.add_argument("--angle-margin-deg", type=float, default=15.0)

    ap.add_argument("--logdir", default="logs_grasp_neg_nowidth")
    ap.add_argument("--ckptdir", default="checkpoints_grasp_neg_nowidth")
    ap.add_argument("--resume", default=None)

    ap.add_argument(
        "--arch",
        default="segformer_b0",
        choices=["resnet_unet", "segformer_b0", "segformer_b2", "swin_tiny"],
        help="Model architecture"
    )

    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--monitor", default="val_loss", choices=["val_loss", "val_acc"])
    ap.add_argument("--pretrained", type=int, default=1)

    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.ckptdir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    include_rgb = bool(args.use_rgb)
    include_depth = bool(args.use_depth)

    # ---- Base file list (stable split) ----
    base_ds = BrickGraspNegDataset(**_make_dataset_kwargs(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        labels_pos_dir=args.labels_pos,
        labels_neg_dir=args.labels_neg,
        output_size=args.output_size,
        include_rgb=include_rgb,
        include_depth=include_depth,
        random_rotate=False,
        random_zoom=False,
        random_translate=False,
        seed=args.seed
    ))
    rgb_files = base_ds.rgb_files

    # ---- Train dataset (with augmentation) ----
    train_ds_full = BrickGraspNegDataset(**_make_dataset_kwargs(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        labels_pos_dir=args.labels_pos,
        labels_neg_dir=args.labels_neg,
        output_size=args.output_size,
        include_rgb=include_rgb,
        include_depth=include_depth,
        random_rotate=True,
        random_zoom=True,
        random_translate=True,
        seed=args.seed,
        rgb_files=rgb_files
    ))

    # ---- Val dataset (NO augmentation) ----
    val_ds_full = BrickGraspNegDataset(**_make_dataset_kwargs(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        labels_pos_dir=args.labels_pos,
        labels_neg_dir=args.labels_neg,
        output_size=args.output_size,
        include_rgb=include_rgb,
        include_depth=include_depth,
        random_rotate=False,
        random_zoom=False,
        random_translate=False,
        seed=args.seed,
        rgb_files=rgb_files
    ))

    n = len(rgb_files)
    n_train = int(0.9 * n)
    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    # Reproducible dataloader randomness
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ---- Model ----
    in_ch = (1 if include_depth else 0) + (3 if include_rgb else 0)
    pretrained = bool(args.pretrained)

    if args.arch == "resnet_unet":
        model = ResNetUNetGraspNoWidth(in_channels=in_ch, pretrained=pretrained).to(device)
    elif args.arch == "segformer_b0":
        model = SegFormerGraspNoWidth(in_channels=in_ch, model_name="segformer_b0", pretrained=pretrained, embed_dim=256).to(device)
    elif args.arch == "segformer_b2":
        model = SegFormerGraspNoWidth(in_channels=in_ch, model_name="segformer_b2", pretrained=pretrained, embed_dim=256).to(device)
    elif args.arch == "swin_tiny":
        model = SwinGraspNoWidth(
            in_channels=in_ch,
            model_name="swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            embed_dim=256,
            img_size=args.output_size,  
        ).to(device)
    else:
        raise ValueError(f"Unknown arch: {args.arch}")

    loss_fn = GraspLossWithNegativesNoWidth(
        w_neg_q=args.w_neg_q,
        w_neg_ang=args.w_neg_ang,
        angle_margin_deg=args.angle_margin_deg
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Resume / early stopping vars ----
    start_epoch = 0
    best_metric = None
    bad_epochs = 0

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_metric = ck.get("best_metric", None)
        print(f"[RESUME] from {args.resume} @ epoch {start_epoch} | best_metric={best_metric}")

    writer = SummaryWriter(args.logdir)
    global_step = 0

    # ---- Freeze backbone ----
    if args.freeze_epochs > 0 and start_epoch < args.freeze_epochs:
        print(f"[FREEZE] backbone frozen for first {args.freeze_epochs} epochs")
        set_backbone_trainable(model, False)

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.freeze_epochs:
            print("[UNFREEZE] backbone is now trainable")
            set_backbone_trainable(model, True)

        model.train()
        t0 = time.time()
        running = 0.0

        for b, (x, y, meta) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = [t.to(device, non_blocking=True) for t in y]

            pred = model(x)
            total, logs = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            running += float(total.item())

            if (b % 20) == 0:
                print(f"[E{epoch:03d}] b={b:04d} loss={total.item():.4f}")

            writer.add_scalar("train/loss", total.item(), global_step)
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", float(v.item()), global_step)
            global_step += 1

        # ---- Validation ----
        model.eval()
        correct = 0
        total_v = 0
        vloss_sum = 0.0

        with torch.no_grad():
            for (xv, yv, meta) in val_loader:
                xv = xv.to(device)
                yv = [t.to(device) for t in yv]

                pred = model(xv)
                vloss, _ = loss_fn(pred, yv)
                vloss_sum += float(vloss.item())

                q, ang = post_process(*pred)

                pos_path = meta["pos_path"][0] if isinstance(meta["pos_path"], (list, tuple)) else meta["pos_path"]
                pos_labels = read_pos_file(pos_path)

                ok = center_angle_match(q, ang, pos_labels)
                correct += int(ok)
                total_v += 1

        acc = correct / max(1, total_v)
        train_loss_epoch = running / max(1, len(train_loader))
        val_loss_epoch = vloss_sum / max(1, total_v)
        dt = time.time() - t0

        print(f"[E{epoch:03d}] train_loss={train_loss_epoch:.4f} val_loss={val_loss_epoch:.4f} val_acc={acc:.3f} time={dt:.1f}s")

        writer.add_scalar("val/acc_center_angle", acc, epoch)
        writer.add_scalar("val/loss", val_loss_epoch, epoch)
        writer.add_scalar("epoch/train_loss", train_loss_epoch, epoch)

        # ---- Save epoch checkpoint ----
        ckpt_path = os.path.join(args.ckptdir, f"{args.arch}_epoch_{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "best_metric": best_metric,
        }, ckpt_path)

        # ---- Early stopping + BEST.pth ----
        current_metric = val_loss_epoch if args.monitor == "val_loss" else acc

        improved = False
        if best_metric is None:
            improved = True
        else:
            if args.monitor == "val_loss":
                improved = (best_metric - current_metric) > args.min_delta  # lower is better
            else:
                improved = (current_metric - best_metric) > args.min_delta  # higher is better

        if improved:
            best_metric = current_metric
            bad_epochs = 0

            best_path = os.path.join(args.ckptdir, "BEST.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "best_metric": best_metric,
            }, best_path)
            print(f"[BEST] saved BEST.pth | {args.monitor}={best_metric:.6f}")
        else:
            bad_epochs += 1
            print(f"[EARLY] no improvement ({bad_epochs}/{args.patience}) on {args.monitor}")

        if bad_epochs >= args.patience:
            print("[STOP] Early stopping triggered.")
            break

    writer.close()


if __name__ == "__main__":
    main()
