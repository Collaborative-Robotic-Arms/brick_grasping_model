import numpy as np
import math


def _affine_params(M):
    """
    M: 2x3 from cv2.getRotationMatrix2D + translation
    returns: rot_rad, scale
    """
    a, b = float(M[0, 0]), float(M[0, 1])
    scale = math.sqrt(a * a + b * b)
    rot = math.atan2(b, a)
    return rot, scale


def _apply_affine_to_label(cx, cy, theta, w, h, M):
    """
    Apply same affine used on the image to label center + angle + size.
    """
    x = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    y = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    rot, scale = _affine_params(M)
    theta2 = theta + rot
    w2 = w * scale
    h2 = h * scale
    return x, y, theta2, w2, h2


def build_targets_from_center_labels(
    pos_list,
    neg_list,
    out_size,
    M_affine=None,
    center_keep=1/3,         # ✅ الثلث الأوسط
    add_edge_as_negative=True,  # ✅ الثلثين بره الوسط يبقوا negative Q
    neg_angle_region=0.25,   # جزء صغير حوالين مركز الـ negative للـ angle-loss
):
    """
    pos_list / neg_list: list of (cx, cy, ang_deg, w, h) in image coords (after resize to out_size)
    Returns:
      pos, cos, sin, neg_mask, neg_cos, neg_sin  each shape (H,W) float32

    Notes:
    - pos is 1 ONLY in the central third of the rectangle (along grasp axis).
    - optional: outer two thirds inside the rectangle are treated as negative Q.
    - cos/sin encode cos(2θ), sin(2θ) so that angle = 0.5 atan2(sin,cos)
    """
    H = W = int(out_size)
    yy, xx = np.mgrid[0:H, 0:W]  # (H,W)

    pos = np.zeros((H, W), dtype=np.float32)
    cos = np.zeros((H, W), dtype=np.float32)
    sin = np.zeros((H, W), dtype=np.float32)

    neg_mask = np.zeros((H, W), dtype=np.float32)
    neg_cos = np.zeros((H, W), dtype=np.float32)
    neg_sin = np.zeros((H, W), dtype=np.float32)

    def _rect_masks(cx, cy, theta, w, h, center_keep_frac):
        """
        Build full rectangle mask + center-third mask (split along u axis).
        u axis = along theta, v axis = perpendicular
        """
        ct = math.cos(theta)
        st = math.sin(theta)

        dx = xx - cx
        dy = yy - cy

        # rotate into grasp frame
        u = dx * ct + dy * st
        v = -dx * st + dy * ct

        full = (np.abs(u) <= (h / 2.0)) & (np.abs(v) <= (w / 2.0))

        # central band along u
        u_th = (h * center_keep_frac) / 2.0
        center = (np.abs(u) <= u_th) & (np.abs(v) <= (w / 2.0))
        center = center & full

        outer = full & (~center)
        return full, center, outer

    # -------- positives --------
    for (cx, cy, ang_deg, w, h) in pos_list:
        theta = math.radians(float(ang_deg))
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)

        if M_affine is not None:
            cx, cy, theta, w, h = _apply_affine_to_label(cx, cy, theta, w, h, M_affine)

        full, center, outer = _rect_masks(cx, cy, theta, w, h, center_keep)

        # positive only in center band
        pos[center] = 1.0
        cos_val = math.cos(2.0 * theta)
        sin_val = math.sin(2.0 * theta)
        cos[center] = cos_val
        sin[center] = sin_val

        # optional: treat edges inside the same rect as negative Q (but NOT angle negative)
        if add_edge_as_negative:
            neg_mask[outer] = 1.0

    # -------- negatives --------
    for (cx, cy, ang_deg, w, h) in neg_list:
        theta = math.radians(float(ang_deg))
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)

        if M_affine is not None:
            cx, cy, theta, w, h = _apply_affine_to_label(cx, cy, theta, w, h, M_affine)

        full, center, _ = _rect_masks(cx, cy, theta, w, h, center_keep)

        # negative Q over full wrong rectangle
        neg_mask[full] = 1.0

        # negative angle region: smaller zone around center (so we don't over-constrain)
        # shrink both axes
        full2, center2, _ = _rect_masks(cx, cy, theta, w * neg_angle_region, h * neg_angle_region, 1.0)
        neg_center = full2  # small blob

        neg_cos[neg_center] = math.cos(2.0 * theta)
        neg_sin[neg_center] = math.sin(2.0 * theta)

    # -------- resolve conflicts: positive wins --------
    pos_idx = pos > 0.5
    neg_mask[pos_idx] = 0.0
    neg_cos[pos_idx] = 0.0
    neg_sin[pos_idx] = 0.0

    return pos, cos, sin, neg_mask, neg_cos, neg_sin
