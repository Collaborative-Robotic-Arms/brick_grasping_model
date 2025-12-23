import numpy as np
import torch


def post_process(pos_logits, cos, sin):
    """
    Inputs torch tensors (B,1,H,W). Return numpy arrays for first item.
    """
    q = torch.sigmoid(pos_logits).detach().cpu().numpy()[0, 0]
    cos = cos.detach().cpu().numpy()[0, 0]
    sin = sin.detach().cpu().numpy()[0, 0]
    ang = 0.5 * np.arctan2(sin, cos)  # [-pi/2, pi/2]
    return q, ang
