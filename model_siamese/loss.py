
from __future__ import annotations

import torch
import torch.nn.functional as F

def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    eps = 1e-6
    ce_pos = -targets * torch.log(p.clamp(min=eps))
    ce_neg = -(1 - targets) * torch.log((1 - p).clamp(min=eps))
    fl_pos = alpha * (1 - p).pow(gamma) * ce_pos
    fl_neg = (1 - alpha) * p.pow(gamma) * ce_neg
    return (fl_pos + fl_neg).mean()

def variance_reg(pooled: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    if pooled.size(0) < 2:
        return pooled.new_zeros(())
    std = pooled.std(dim=0, unbiased=False)
    return F.relu(target - std).mean()

def decorrelation_reg(pooled: torch.Tensor) -> torch.Tensor:
    B, D = pooled.shape
    if B < 2:
        return pooled.new_zeros(())
    p = pooled - pooled.mean(dim=0, keepdim=True)
    p = p / (p.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    corr = (p.t() @ p) / (B - 1)
    eye = torch.eye(D, device=corr.device, dtype=corr.dtype)
    off = corr - eye
    return (off ** 2).mean()

def total_loss(
    out: dict[str, torch.Tensor],
    is_present: torch.Tensor,
    *,
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
    variance_target: float = 0.5,
    variance_weight: float = 0.1,
    decorr_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    targets = is_present.float()
    logit = out["existence_logit"]
    pooled = out["pooled"]
    focal = focal_bce_loss(logit, targets, focal_alpha, focal_gamma)
    var = variance_reg(pooled, target=variance_target)
    decor = decorrelation_reg(pooled)
    total = focal + variance_weight * var + decorr_weight * decor
    return {
        "loss": total, "focal": focal, "variance": var, "decorrelation": decor,
    }
