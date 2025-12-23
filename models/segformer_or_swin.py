import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

try:
    from transformers import SegformerModel, SegformerConfig
except Exception:
    SegformerModel = None
    SegformerConfig = None


def _resolve_mit_hf_id(name: str) -> str:
    """
    Map your CLI names to HF model ids.
    You can also pass a full HF id directly (e.g. "nvidia/mit-b2").
    """
    n = (name or "").strip().lower()

    mapping = {
        "segformer_b0": "nvidia/mit-b0",
        "mit_b0": "nvidia/mit-b0",
        "b0": "nvidia/mit-b0",

        "segformer_b1": "nvidia/mit-b1",
        "mit_b1": "nvidia/mit-b1",
        "b1": "nvidia/mit-b1",

        "segformer_b2": "nvidia/mit-b2",
        "mit_b2": "nvidia/mit-b2",
        "b2": "nvidia/mit-b2",

        "segformer_b3": "nvidia/mit-b3",
        "mit_b3": "nvidia/mit-b3",
        "b3": "nvidia/mit-b3",

        "segformer_b4": "nvidia/mit-b4",
        "mit_b4": "nvidia/mit-b4",
        "b4": "nvidia/mit-b4",

        "segformer_b5": "nvidia/mit-b5",
        "mit_b5": "nvidia/mit-b5",
        "b5": "nvidia/mit-b5",
    }

    return mapping.get(n, name)  # allow passing full HF id directly


class SegFormerGraspNoWidth(nn.Module):
    """
    SegFormer(MiT) encoder from HuggingFace Transformers, + lightweight fusion decoder.
    Outputs: (pos_logits, cos, sin) with same H,W as input.
    """
    def __init__(
        self,
        in_channels: int,
        model_name: str = "segformer_b0",   # you pass segformer_b0 in train.py
        pretrained: bool = True,
        embed_dim: int = 256,
    ):
        super().__init__()

        if SegformerModel is None:
            raise ImportError(
                "transformers is not installed. Run:\n"
                "  conda install -c conda-forge transformers huggingface_hub safetensors"
            )

        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)

        # If your input is 4-ch (RGBD), project to 3-ch so we can reuse pretrained MiT weights.
        if self.in_channels == 3:
            self.pre = nn.Identity()
        else:
            self.pre = nn.Conv2d(self.in_channels, 3, kernel_size=1, bias=False)

        hf_id = _resolve_mit_hf_id(model_name)

        if pretrained:
            self.backbone = SegformerModel.from_pretrained(hf_id)
        else:
            cfg = SegformerConfig()
            self.backbone = SegformerModel(cfg)

        # make sure we get feature maps (hidden_states)
        self.backbone.config.output_hidden_states = True

        # MiT stage channel dims (e.g. b0: [32, 64, 160, 256])
        hs = list(self.backbone.config.hidden_sizes)

        self.proj = nn.ModuleList([nn.Conv2d(c, self.embed_dim, kernel_size=1) for c in hs])

        # fuse multi-scale to highest-res stage (H/4)
        self.fuse = nn.Sequential(
            nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head_pos = nn.Conv2d(self.embed_dim, 1, kernel_size=1)
        self.head_cos = nn.Conv2d(self.embed_dim, 1, kernel_size=1)
        self.head_sin = nn.Conv2d(self.embed_dim, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: pos_logits, cos, sin each (B,1,H,W)
        """
        B, C, H, W = x.shape

        pv = self.pre(x)

        out = self.backbone(pv, output_hidden_states=True, return_dict=True)

        # out.hidden_states: (embeddings + each stage) feature maps
        # take 4 stages
        feats = list(out.hidden_states[1:])  # 4 tensors: (B, Ci, Hi, Wi)

        # project to embed_dim
        p = [proj(f) for proj, f in zip(self.proj, feats)]

        # upsample all to stage0 resolution (usually H/4, W/4)
        target_hw = p[0].shape[-2:]
        p_up = [p[0]]
        for i in range(1, 4):
            p_up.append(F.interpolate(p[i], size=target_hw, mode="bilinear", align_corners=False))

        fused = self.fuse(torch.cat(p_up, dim=1))

        # upsample to input size
        fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)

        pos = self.head_pos(fused)
        cos = self.head_cos(fused)
        sin = self.head_sin(fused)
        return pos, cos, sin


class SwinGraspNoWidth(nn.Module):
    """
    Swin backbone via timm + fusion + heads.
    IMPORTANT: set img_size to match your input (e.g. 160).
    """
    def __init__(
        self,
        in_channels: int,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        embed_dim: int = 256,
        out_indices=(0, 1, 2, 3),
        img_size: int = 160,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=out_indices,
            img_size=img_size,          # ✅ أهم سطر
        )

        chs = self.backbone.feature_info.channels()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in chs])

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(chs), embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head_pos = nn.Conv2d(embed_dim, 1, 1)
        self.head_cos = nn.Conv2d(embed_dim, 1, 1)
        self.head_sin = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.backbone(x)

        p = []
        for proj, f in zip(self.proj, feats):
            # timm Swin sometimes returns NHWC -> convert to NCHW for Conv2d
            if f.ndim == 4 and (f.shape[1] != proj.in_channels) and (f.shape[-1] == proj.in_channels):
                f = f.permute(0, 3, 1, 2).contiguous()
            p.append(proj(f))

        
        target_hw = p[0].shape[-2:]

        p_up = [p[0]]
        for i in range(1, len(p)):
            p_up.append(F.interpolate(p[i], size=target_hw, mode="bilinear", align_corners=False))

        fused = self.fuse(torch.cat(p_up, dim=1))
        fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)

        pos = self.head_pos(fused)
        cos = self.head_cos(fused)
        sin = self.head_sin(fused)
        return pos, cos, sin

