"""Backbone + grading head.

Primary backbone is **RETFound** (ViT-Large/16, MAE-pretrained on ~1.6M retinal
images). RETFound publishes encoder weights compatible with a timm ViT-L. We
build a timm ``vit_large_patch16_224`` with mean (``avg``) global pooling -- the
configuration used in RETFound's fine-tuning recipe -- and load the published
checkpoint with a tolerant key-matching routine (handles ``pos_embed``
interpolation and drops decoder/mask keys).

If no checkpoint is supplied we fall back to ImageNet-pretrained weights (the
``vit_l_imagenet`` ablation backbone) so the pipeline is always runnable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn

from .config import NUM_CLASSES


def build_model(backbone: str, num_classes: int = NUM_CLASSES, drop_path: float = 0.2) -> nn.Module:
    """Create a ViT-L/16@224 with mean pooling and a ``num_classes`` head."""
    pretrained = backbone == "vit_l_imagenet"
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        global_pool="avg",
        drop_path_rate=drop_path,
    )
    return model


@torch.no_grad()
def _interpolate_pos_embed(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Interpolate positional embeddings if the checkpoint grid differs."""
    if "pos_embed" not in state or not hasattr(model, "pos_embed"):
        return
    ckpt_pe = state["pos_embed"]
    model_pe = model.pos_embed
    if ckpt_pe.shape == model_pe.shape:
        return
    # tokens = grid*grid + num_prefix (cls). Infer prefix from model.
    num_prefix = model_pe.shape[1] - (model.patch_embed.num_patches)
    ckpt_grid = int((ckpt_pe.shape[1] - num_prefix) ** 0.5)
    model_grid = int(model.patch_embed.num_patches ** 0.5)
    if ckpt_grid == model_grid:
        return
    prefix = ckpt_pe[:, :num_prefix]
    grid = ckpt_pe[:, num_prefix:].reshape(1, ckpt_grid, ckpt_grid, -1).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid, size=(model_grid, model_grid), mode="bicubic", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, model_grid * model_grid, -1)
    state["pos_embed"] = torch.cat([prefix, grid], dim=1)


def load_retfound_checkpoint(model: nn.Module, ckpt_path: str) -> Dict:
    """Load RETFound weights tolerantly. Returns a load report (matched/missing)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Normalise common prefixes and drop non-encoder / head keys.
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "encoder.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        if nk.startswith("decoder") or "mask_token" in nk:
            continue
        if nk.startswith("head.") or nk in ("head.weight", "head.bias"):
            continue  # re-initialise the classification head for 5 classes
        cleaned[nk] = v

    _interpolate_pos_embed(model, cleaned)
    result = model.load_state_dict(cleaned, strict=False)
    report = {
        "ckpt_path": ckpt_path,
        "n_ckpt_tensors": len(state),
        "n_loaded": len(cleaned) - len(result.unexpected_keys),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
    return report


def create_backbone(cfg) -> Tuple[nn.Module, Dict]:
    """Build model per TrainConfig and load weights. Returns (model, load_report)."""
    model = build_model(cfg.backbone, NUM_CLASSES, cfg.drop_path)
    report: Dict = {"backbone": cfg.backbone}
    if cfg.backbone == "retfound":
        if cfg.retfound_ckpt:
            report.update(load_retfound_checkpoint(model, cfg.retfound_ckpt))
        else:
            report["warning"] = (
                "No RETFOUND_CKPT provided; backbone is randomly initialised. "
                "Set the checkpoint path for the real RETFound experiment."
            )
    return model, report


def param_groups_lrd(
    model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float
) -> List[Dict]:
    """Layer-wise LR decay groups (RETFound/MAE fine-tuning recipe).

    Earlier transformer blocks get exponentially smaller LRs than later ones.
    Norm/bias/1-D params receive no weight decay.
    """
    # Determine per-parameter "layer id": patch_embed/cls/pos -> 0,
    # block i -> i+1, head/norm -> num_blocks+1.
    blocks = getattr(model, "blocks", None)
    num_layers = (len(blocks) if blocks is not None else 0) + 1

    def layer_id(name: str) -> int:
        if name in ("cls_token", "pos_embed") or name.startswith("patch_embed"):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return num_layers

    scales = {i: layer_decay ** (num_layers - i) for i in range(num_layers + 1)}
    groups: Dict[str, Dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        no_decay = p.ndim == 1 or name.endswith(".bias")
        key = f"layer{lid}_{'nd' if no_decay else 'wd'}"
        if key not in groups:
            groups[key] = {
                "lr": base_lr * scales[lid],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "params": [],
            }
        groups[key]["params"].append(p)
    return list(groups.values())
