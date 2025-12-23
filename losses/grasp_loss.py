import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspLossWithNegativesNoWidth(nn.Module):
    """
    Total = Q loss + angle regression (pos-only) + neg Q penalty + neg angle penalty

    تعديل مهم:
    - neg angle penalty يتحسب فقط على pixels اللي عندها neg_cos/neg_sin (يعني negative-angle targets موجودة فعلاً)
      عشان edge-negative (بتاع تقسيم المستطيل) ما يـ dilute الـ neg_ang_loss.
    """
    def __init__(self, w_neg_q=0.5, w_neg_ang=0.2, angle_margin_deg=15.0):
        super().__init__()
        self.w_neg_q = float(w_neg_q)
        self.w_neg_ang = float(w_neg_ang)
        self.margin = math.radians(float(angle_margin_deg))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, target):
        """
        pred:   (pos_logits, cos2, sin2)
        target: (pos, cos, sin, neg_mask, neg_cos, neg_sin)
        All B,1,H,W
        """
        pos_logits, cos_p, sin_p = pred
        pos_t, cos_t, sin_t, neg_m, neg_cos_t, neg_sin_t = target

        eps = 1e-6

        # --- Q loss (positive center band only because pos_t is 1 only there) ---
        pos_loss_map = self.bce(pos_logits, pos_t)
        pos_loss = pos_loss_map.mean()

        # --- angle regression (only where positive) ---
        m_pos = pos_t.detach()
        den_pos = m_pos.sum().clamp(min=1.0)
        cos_loss = (self.smooth(cos_p, cos_t) * m_pos).sum() / den_pos
        sin_loss = (self.smooth(sin_p, sin_t) * m_pos).sum() / den_pos

        # --- negative Q penalty (don’t punish where positive) ---
        # neg_m هنا ممكن يشمل: true-negative rects + edge-negative (outer thirds)
        neg_area_q = (neg_m * (1.0 - pos_t)).detach()
        den_nq = neg_area_q.sum().clamp(min=1.0)
        neg_q_map = self.bce(pos_logits, torch.zeros_like(pos_t))
        neg_q_loss = (neg_q_map * neg_area_q).sum() / den_nq

        # --- negative angle penalty (ONLY where neg angle targets exist) ---
        # detect where neg_cos/sin are actually defined (non-zero)
        has_neg_ang = ((neg_cos_t.abs() + neg_sin_t.abs()) > 1e-8).float()

        # normalize predicted angle vector
        pnorm = torch.sqrt(cos_p**2 + sin_p**2 + eps)
        pc = cos_p / pnorm
        ps = sin_p / pnorm

        # normalize negative target vector
        tnorm = torch.sqrt(neg_cos_t**2 + neg_sin_t**2 + eps)
        tc = neg_cos_t / tnorm
        ts = neg_sin_t / tnorm

        sim = pc * tc + ps * ts  # cos(2Δ)
        thr = math.cos(2.0 * self.margin)

        # IMPORTANT: use mask that has neg angle targets only
        neg_area_ang = (neg_m * has_neg_ang).detach()
        den_na = neg_area_ang.sum().clamp(min=1.0)
        neg_ang_loss = (F.relu(sim - thr) * neg_area_ang).sum() / den_na

        total = pos_loss + cos_loss + sin_loss \
                + (self.w_neg_q * neg_q_loss) \
                + (self.w_neg_ang * neg_ang_loss)

        return total, {
            "pos_loss": pos_loss.detach(),
            "cos_loss": cos_loss.detach(),
            "sin_loss": sin_loss.detach(),
            "neg_q_loss": neg_q_loss.detach(),
            "neg_ang_loss": neg_ang_loss.detach(),
        }
