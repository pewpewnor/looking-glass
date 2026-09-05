
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Owlv2ForObjectDetection

OWLV2_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWLV2_STD = (0.26862954, 0.26130258, 0.27577711)

OWLV2_MODEL_NAME = "google/owlv2-base-patch16-ensemble"

def _normalize_owlv2(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(OWLV2_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(OWLV2_STD).view(1, 3, 1, 1)
    return (x - mean) / std

class SupportAttnPool(nn.Module):

    def __init__(self, d_v: int, d_q: int, n_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        assert d_q % n_heads == 0, "d_q must be divisible by n_heads"
        self.d_v = d_v
        self.d_q = d_q
        self.n_heads = n_heads
        self.proj_v = nn.Linear(d_v, d_q)
        self.proj_q = nn.Linear(d_q, d_q)
        self.proj_k = nn.Linear(d_q, d_q)
        self.proj_o = nn.Linear(d_q, d_q)
        self.norm_q = nn.LayerNorm(d_q)
        self.norm_kv = nn.LayerNorm(d_q)
        # Residual gate: the q_emb path is the baseline; the pool adds a
        # learned correction. Starts at 0.1 so training warm-starts close
        # to the bare-q_emb prototype.
        self.residual_gate = nn.Parameter(torch.full((), 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feats: torch.Tensor,
        q_emb: torch.Tensor,
    ) -> torch.Tensor:
        BK, P, _ = feats.shape
        D_q = self.d_q
        H = self.n_heads
        Dh = D_q // H

        proj = self.proj_v(feats)
        kv_in = self.norm_kv(proj)
        q_in  = self.norm_q(q_emb).unsqueeze(1)

        q = self.proj_q(q_in).view(BK, 1, H, Dh).transpose(1, 2)
        k = self.proj_k(kv_in).view(BK, P, H, Dh).transpose(1, 2)
        v = kv_in.view(BK, P, H, Dh).transpose(1, 2)

        scale = 1.0 / math.sqrt(Dh)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(BK, 1, D_q)
        pooled = self.proj_o(out).squeeze(1)
        return q_emb + self.residual_gate * pooled

def _centre_gaussian_bias_DEPRECATED(gh: int, gw: int, sigma_frac: float,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = (torch.arange(gh, device=device, dtype=dtype) + 0.5) / gh - 0.5
    x = (torch.arange(gw, device=device, dtype=dtype) + 0.5) / gw - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = yy * yy + xx * xx
    sigma2 = max(1e-4, sigma_frac) ** 2
    return (-0.5 * r2 / sigma2).reshape(-1)

class LogScaleBoxHead(nn.Module):

    def __init__(self, w_min: float = 0.005, w_max: float = 1.0) -> None:
        super().__init__()
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        # Per-axis log-scale parameters. Initialised so that at start, the
        # output is identical to the underlying box_predictor.
        self.log_w_scale = nn.Parameter(torch.zeros(()))
        self.log_h_scale = nn.Parameter(torch.zeros(()))
        # Per-axis log-bias so the head can shift mean predicted area
        # without needing the OWLv2 box_predictor to update.
        self.log_w_bias = nn.Parameter(torch.zeros(()))
        self.log_h_bias = nn.Parameter(torch.zeros(()))

    def forward(self, raw_boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = raw_boxes.unbind(-1)
        # Convert raw widths to log-space, scale, bias, then back to width.
        # softplus keeps the gradient alive for tiny raw widths.
        log_w = F.softplus(w + 1e-3).log() if False else (w + 1e-3).clamp(min=1e-4).log()
        log_h = (h + 1e-3).clamp(min=1e-4).log()
        log_w = log_w * (1.0 + self.log_w_scale) + self.log_w_bias
        log_h = log_h * (1.0 + self.log_h_scale) + self.log_h_bias
        new_w = log_w.exp().clamp(min=self.w_min, max=self.w_max)
        new_h = log_h.exp().clamp(min=self.w_min, max=self.w_max)
        return torch.stack([cx, cy, new_w, new_h], dim=-1)

class MultiShotLocalizer(nn.Module):

    def __init__(
        self,
        model_name: str = OWLV2_MODEL_NAME,
        *,
        k_max: int = 10,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        fusion_mlp_ratio: int = 2,
        fusion_dropout: float = 0.1,

        support_attn_heads: int = 4,
        support_attn_dropout: float = 0.0,

        alpha_init: float = 1.0,

        use_log_box_head: bool = True,
        log_box_w_min: float = 0.005,
    ) -> None:
        super().__init__()
        self.owlv2 = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.k_max = int(k_max)
        D_q = self.owlv2.config.text_config.hidden_size
        D_v = self.owlv2.config.vision_config.hidden_size
        self.query_dim = D_q
        self.vision_dim = D_v

        self.support_pool = SupportAttnPool(
            d_v=D_v, d_q=D_q,
            n_heads=int(support_attn_heads),
            dropout=float(support_attn_dropout),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, D_q) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_q, nhead=fusion_heads,
            dim_feedforward=D_q * fusion_mlp_ratio,
            dropout=fusion_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=fusion_layers, enable_nested_tensor=False,
        )
        self.fusion_norm = nn.LayerNorm(D_q)

        # Residual gate alpha. With ``alpha_init=1.0`` the prototype starts
        # as ``baseline + 1.0 * (fused - baseline) = fused``, so the fusion
        # path is fully on. The previous 0.01 init kept the prototype pinned
        # at the baseline (mean of q_emb) for the early epochs.
        self.alpha = nn.Parameter(torch.full((), float(alpha_init)))

        self.bg_prototype = nn.Parameter(torch.zeros(1, D_q))
        nn.init.normal_(self.bg_prototype, std=0.02)
        self.bg_bias = nn.Parameter(torch.zeros(()))

        self.use_log_box_head = bool(use_log_box_head)
        self.log_box_head = LogScaleBoxHead(w_min=log_box_w_min) if self.use_log_box_head else None

        self.freeze_backbone()
        self._lora_attached = False

    def freeze_backbone(self) -> None:
        for p in self.owlv2.parameters():
            p.requires_grad = False

    def unfreeze_heads(self) -> None:
        for p in self.owlv2.class_head.parameters():
            p.requires_grad = True
        for p in self.owlv2.box_head.parameters():
            p.requires_grad = True
        for p in self.owlv2.layer_norm.parameters():
            p.requires_grad = True

    def freeze_box_head(self) -> None:
        for p in self.owlv2.box_head.parameters():
            p.requires_grad = False

    def unfreeze_box_head(self) -> None:
        for p in self.owlv2.box_head.parameters():
            p.requires_grad = True

    def class_head_params(self) -> list[nn.Parameter]:
        return list(self.owlv2.class_head.parameters()) + list(self.owlv2.layer_norm.parameters())

    def box_head_params(self) -> list[nn.Parameter]:
        out = list(self.owlv2.box_head.parameters())
        if self.log_box_head is not None:
            out += list(self.log_box_head.parameters())
        return out

    def fusion_params(self) -> list[nn.Parameter]:
        return (
            list(self.fusion.parameters())
            + [self.cls_token, self.alpha, self.bg_prototype, self.bg_bias]
            + list(self.fusion_norm.parameters())
            + list(self.support_pool.parameters())
        )

    def attach_lora(
        self, *, r: int = 8, alpha: int = 16, dropout: float = 0.1, last_n_layers: int = 4,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> list[nn.Parameter]:
        if self._lora_attached:
            lora_params = [
                p for n, p in self.owlv2.named_parameters()
                if "lora_" in n and p.requires_grad
            ]
            if not lora_params:
                lora_params = [p for n, p in self.owlv2.named_parameters() if "lora_" in n]
            return lora_params

        from peft import LoraConfig, get_peft_model

        encoder_layers = self.owlv2.owlv2.vision_model.encoder.layers
        n_layers = len(encoder_layers)
        target_layer_ids = list(range(max(0, n_layers - last_n_layers), n_layers))
        target_module_paths = [
            f"owlv2.vision_model.encoder.layers.{i}.self_attn.{proj}"
            for i in target_layer_ids
            for proj in target_modules
        ]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_module_paths, bias="none",
        )
        self.owlv2 = get_peft_model(self.owlv2, cfg)
        self._lora_attached = True
        return [p for n, p in self.owlv2.named_parameters() if "lora_" in n and p.requires_grad]

    @property
    def lora_attached(self) -> bool:
        return self._lora_attached

    def _support_pass(
        self, support_imgs: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        if support_imgs.dim() != 5:
            raise ValueError(
                f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}"
            )
        B, K, _, S1, S2 = support_imgs.shape
        flat = support_imgs.reshape(B * K, 3, S1, S2)
        flat = _normalize_owlv2(flat)

        # OWLv2 backbone path. Run under no_grad when no class_head param is
        # trainable AND no LoRA on the vision tower is trainable.
        any_grad = torch.is_grad_enabled() and (
            any(p.requires_grad for p in self.owlv2.class_head.parameters())
            or any("lora_" in n and p.requires_grad
                   for n, p in self.owlv2.named_parameters())
        )
        ctx = torch.enable_grad() if any_grad else torch.no_grad()
        with ctx:
            fm, _ = self.owlv2.image_embedder(
                pixel_values=flat, interpolate_pos_encoding=True,
            )
            gh, gw = fm.shape[1], fm.shape[2]
            feats = fm.reshape(B * K, -1, fm.shape[-1])
            q_emb, _, _ = self.owlv2.embed_image_query(
                feats, fm, interpolate_pos_encoding=True,
            )

        if q_emb.dim() == 3:
            q_emb = q_emb.squeeze(1)

        # No centre prior — supports are pre-cropped to the object on disk
        # since manifest v5, so every patch is object signal.
        with ctx:
            pooled = self.support_pool(feats, q_emb)

        out = pooled.view(B, K, -1)
        return out if any_grad else out.detach()

    @staticmethod
    def _baseline_prototype(
        q_emb: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = support_mask.float().unsqueeze(-1)
        return (q_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _fuse(
        self, q_emb: torch.Tensor, support_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, K, _ = q_emb.shape
        if (~support_mask).all(dim=1).any():
            support_mask = support_mask.clone()
            empty_rows = (~support_mask).all(dim=1)
            support_mask[empty_rows, 0] = True
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, q_emb], dim=1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        kp_mask = torch.cat([cls_mask, ~support_mask], dim=1)
        fused = self.fusion(seq, src_key_padding_mask=kp_mask)
        return self.fusion_norm(fused[:, 0])

    def _class_predict(
        self, feats_q: torch.Tensor, proto: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.owlv2.class_predictor(feats_q, proto.unsqueeze(1))
        return logits.squeeze(-1)

    def _decode_boxes(self, feats_q: torch.Tensor, fm_q: torch.Tensor) -> torch.Tensor:
        raw = self.owlv2.box_predictor(
            feats_q, fm_q, interpolate_pos_encoding=True,
        )
        if self.log_box_head is not None:
            raw = self.log_box_head(raw)
        return raw

    def forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if support_mask.dtype != torch.bool:
            support_mask = support_mask.to(torch.bool)
        if support_mask.device != support_imgs.device:
            support_mask = support_mask.to(support_imgs.device)

        q_emb = self._support_pass(support_imgs, support_mask)
        baseline = self._baseline_prototype(q_emb, support_mask)
        fused = self._fuse(q_emb, support_mask)
        # Residual: prototype = baseline + alpha * (fused - baseline).
        # When fused ≈ baseline the prototype ≈ baseline regardless of α; when
        # fused diverges from baseline, α controls how much divergence the
        # prototype keeps. alpha_init=1.0 ⇒ start from the fused path.
        prototype = baseline + self.alpha * (fused - baseline)

        q_norm = _normalize_owlv2(query_img)
        fm_q, _ = self.owlv2.image_embedder(pixel_values=q_norm, interpolate_pos_encoding=True)
        gh, gw = fm_q.shape[1], fm_q.shape[2]
        feats_q = fm_q.reshape(fm_q.shape[0], -1, fm_q.shape[-1])
        B = feats_q.shape[0]

        pred_logits_fg = self._class_predict(feats_q, prototype)
        bg_proto_b = self.bg_prototype.expand(B, -1)
        bg_logits_patch = self._class_predict(feats_q, bg_proto_b)
        bg_logit = bg_logits_patch.max(dim=-1).values + self.bg_bias

        pred_boxes = self._decode_boxes(feats_q, fm_q)

        joint = torch.cat([pred_logits_fg, bg_logit.unsqueeze(-1)], dim=-1)
        joint_prob = joint.softmax(dim=-1)
        fg_prob = joint_prob[:, :-1]
        bg_prob = joint_prob[:, -1]
        best_idx = fg_prob.argmax(dim=-1)
        ar = torch.arange(B, device=pred_logits_fg.device)
        best_box = pred_boxes[ar, best_idx]
        best_score = fg_prob[ar, best_idx]
        best_logit = pred_logits_fg[ar, best_idx]

        return {
            "best_box": best_box,
            "best_score": best_score,
            "bg_prob": bg_prob,
            "best_logit": best_logit,
            "pred_logits": pred_logits_fg,
            "pred_logits_fg": pred_logits_fg,
            "bg_logit": bg_logit,
            "bg_logits_patch": bg_logits_patch,
            "pred_boxes": pred_boxes,
            "prototype": prototype,
            "bg_prototype": self.bg_prototype,
            "baseline_prototype": baseline,
            "alpha": self.alpha.detach(),
            "patch_grid": (gh, gw),
        }

    @torch.no_grad()
    def phase0_forward(
        self,
        support_imgs: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if support_imgs.dim() != 5:
            raise ValueError(f"support_imgs must be (B, K, 3, S, S), got {tuple(support_imgs.shape)}")
        B, K, _, S1, S2 = support_imgs.shape
        # Pick the first VALID slot per row. ``support_mask`` is True on real
        # supports. Rows that are entirely padded get slot 0 anyway (and
        # match the trained forward's defensive behaviour).
        safe_mask = support_mask
        if (~safe_mask).all(dim=1).any():
            safe_mask = safe_mask.clone()
            safe_mask[(~safe_mask).all(dim=1), 0] = True
        first_idx = safe_mask.float().argmax(dim=1)
        ar = torch.arange(B, device=support_imgs.device)
        support_one = support_imgs[ar, first_idx]

        flat = _normalize_owlv2(support_one)
        fm, _ = self.owlv2.image_embedder(pixel_values=flat, interpolate_pos_encoding=True)
        feats = fm.reshape(B, -1, fm.shape[-1])
        q_emb, _, _ = self.owlv2.embed_image_query(feats, fm, interpolate_pos_encoding=True)
        if q_emb.dim() == 3:
            q_emb = q_emb.squeeze(1)
        proto = q_emb

        q_norm = _normalize_owlv2(query_img)
        fm_q, _ = self.owlv2.image_embedder(pixel_values=q_norm, interpolate_pos_encoding=True)
        feats_q = fm_q.reshape(fm_q.shape[0], -1, fm_q.shape[-1])
        pred_logits, _ = self.owlv2.class_predictor(feats_q, proto.unsqueeze(1))
        pred_logits = pred_logits.squeeze(-1)

        pred_boxes = self.owlv2.box_predictor(feats_q, fm_q, interpolate_pos_encoding=True)
        best_idx = pred_logits.argmax(dim=-1)
        ar2 = torch.arange(B, device=pred_logits.device)
        return {
            "best_box": pred_boxes[ar2, best_idx],
            "best_score": torch.sigmoid(pred_logits[ar2, best_idx]),
            "bg_prob": torch.zeros(B, device=pred_logits.device),
            "best_logit": pred_logits[ar2, best_idx],
            "pred_logits": pred_logits,
            "pred_logits_fg": pred_logits,
            "bg_logit": torch.zeros(B, device=pred_logits.device),
            "bg_logits_patch": torch.zeros_like(pred_logits),
            "pred_boxes": pred_boxes,
            "prototype": proto,
            "bg_prototype": torch.zeros(1, proto.size(-1), device=proto.device),
            "baseline_prototype": proto,
            "alpha": torch.zeros((), device=proto.device),
            "patch_grid": (fm_q.shape[1], fm_q.shape[2]),
        }
