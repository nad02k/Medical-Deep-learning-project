"""Multimodal fusion modules for chest X-ray + EHR data.

Architecture
============
- **Image encoder** : Pre-trained ResNet-50 → spatial feature tokens projected
  to *d*-dimensional space.
- **EHR encoder** : Learnable feature embeddings + positional encoding →
  2-layer Transformer encoder → projected to *d*-dimensional space.
- **Fusion** : Bidirectional cross-attention with gating
  (image queries attend to EHR keys/values *and* vice-versa).
- **Prediction head** : MLP for multi-label pathology classification (14
  classes by default — CheXpert / NIH labelling convention).

Fusion strategy
---------------
Intermediate fusion with cross-attention is chosen because it allows
fine-grained interaction between local visual features (spatial regions of
the chest X-ray) and individual clinical measurements, rather than collapsing
each modality to a single vector before combining them.

Pseudo-algorithm (Task 3 spec)::

    h_A ← Proj_A(E_A(x_A))              # image features
    h_B ← Proj_B(E_B(x_B))              # EHR features
    if mask_B == 0: h_B ← learned_absent_token
    z   ← F(h_A, h_B)                   # cross-attention fusion
    ŷ   ← H(z)                          # classification head
    L   ← L_task(ŷ, y) + λ · L_aux      # BCE + optional unimodal aux losses
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_D = 256          # shared embedding dimension
_DEFAULT_HEADS = 8        # attention heads
_DEFAULT_N_CLASSES = 14   # pathology labels
_DEFAULT_EHR_FEATURES = 64  # number of tabular EHR input features
_DEFAULT_IMG_SIZE = 224   # expected input spatial size


# ──────────────────────────────────────────────────────────────────────
# Image Encoder
# ──────────────────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """ResNet-50 backbone → spatial feature tokens projected to *d* dims.

    Input
    -----
    ``(B, 1, H, W)`` — single-channel (grayscale) chest X-ray images.

    Output
    ------
    ``(B, N, d)`` — *N* spatial feature tokens where ``N = (H/32) * (W/32)``
    for the default ResNet-50 stride-32 feature map.
    """

    def __init__(
        self,
        d: int = _DEFAULT_D,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load pre-trained ResNet-50 and adapt first conv for 1-channel input.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Replace first conv: 3-channel → 1-channel (average pretrained
        # weights).
        original_conv = resnet.conv1
        self.input_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,  # type: ignore[arg-type]
            stride=original_conv.stride,            # type: ignore[arg-type]
            padding=original_conv.padding,          # type: ignore[arg-type]
            bias=False,
        )
        if pretrained:
            # Average the 3-channel weights into 1 channel.
            with torch.no_grad():
                self.input_conv.weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Keep layers up to the last convolutional block (layer4).
        self.backbone = nn.Sequential(
            self.input_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection from ResNet feature dim (2048) to shared dim d.
        self.proj = nn.Sequential(
            nn.Linear(2048, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, 1, H, W)``

        Returns
        -------
        Tensor of shape ``(B, N, d)`` where ``N = h * w`` spatial tokens.
        """
        feat_map = self.backbone(x)  # (B, 2048, h, w)
        B, C, h, w = feat_map.shape
        tokens = feat_map.reshape(B, C, h * w).permute(0, 2, 1)  # (B, N, 2048)
        return self.proj(tokens)  # (B, N, d)


# ──────────────────────────────────────────────────────────────────────
# EHR Encoder
# ──────────────────────────────────────────────────────────────────────

class EHREncoder(nn.Module):
    """Tabular EHR feature encoder using a Transformer.

    Each input feature is individually embedded, positional encoding is
    added, and the resulting sequence is processed by a 2-layer Transformer
    encoder.  The output is a set of *M* contextualized tokens in
    *d*-dimensional space (``M = num_features``).

    Input
    -----
    ``(B, num_features)`` — flat tabular EHR vector.

    Output
    ------
    ``(B, M, d)`` — contextualized EHR tokens.
    """

    def __init__(
        self,
        num_features: int = _DEFAULT_EHR_FEATURES,
        d: int = _DEFAULT_D,
        n_heads: int = _DEFAULT_HEADS,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.d = d

        # Per-feature linear embedding.
        self.feature_embed = nn.Linear(1, d)

        # Learnable positional encoding (one per feature slot).
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_features, d) * 0.02)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, num_features)``

        Returns
        -------
        Tensor of shape ``(B, M, d)`` with ``M = num_features``.
        """
        B = x.size(0)
        # (B, M, 1) → (B, M, d)
        tokens = self.feature_embed(x.unsqueeze(-1))
        tokens = tokens + self.pos_encoding[:, : self.num_features, :]
        tokens = self.transformer(tokens)
        return self.layer_norm(tokens)


# ──────────────────────────────────────────────────────────────────────
# Cross-Attention Fusion
# ──────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention with gating for multimodal fusion.

    1. Image queries attend to EHR keys/values → ``img_cross``.
    2. EHR queries attend to image keys/values → ``ehr_cross``.
    3. Both are pooled (mean), concatenated, and gated through an MLP
       to produce a single fused representation.

    Input
    -----
    ``h_img`` : ``(B, N, d)`` — image tokens.
    ``h_ehr`` : ``(B, M, d)`` — EHR tokens.

    Output
    ------
    ``(B, d)`` — fused representation.
    """

    def __init__(
        self,
        d: int = _DEFAULT_D,
        n_heads: int = _DEFAULT_HEADS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Image queries → EHR context.
        self.cross_attn_img = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_img = nn.LayerNorm(d)

        # EHR queries → Image context.
        self.cross_attn_ehr = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ehr = nn.LayerNorm(d)

        # Gating MLP: 2*d → d.
        self.gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

    def forward(
        self,
        h_img: torch.Tensor,
        h_ehr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        h_img : ``(B, N, d)``
        h_ehr : ``(B, M, d)``

        Returns
        -------
        ``(B, d)`` — gated fused representation.
        """
        # Bidirectional cross-attention.
        img_cross, _ = self.cross_attn_img(
            query=h_img, key=h_ehr, value=h_ehr
        )
        img_cross = self.norm_img(h_img + img_cross)  # residual

        ehr_cross, _ = self.cross_attn_ehr(
            query=h_ehr, key=h_img, value=h_img
        )
        ehr_cross = self.norm_ehr(h_ehr + ehr_cross)  # residual

        # Mean pool each branch → (B, d).
        img_pooled = img_cross.mean(dim=1)
        ehr_pooled = ehr_cross.mean(dim=1)

        # Concatenate + gate.
        concat = torch.cat([img_pooled, ehr_pooled], dim=-1)  # (B, 2d)
        gate = self.gate(concat)   # (B, d)
        proj = self.proj(concat)   # (B, d)

        return gate * proj  # (B, d)


# ──────────────────────────────────────────────────────────────────────
# Modality Dropout
# ──────────────────────────────────────────────────────────────────────

class ModalityDropout(nn.Module):
    """Randomly zero out one modality during training for robustness.

    With probability ``p_drop`` each modality is independently dropped
    (zeroed), but at least one modality is always kept.

    Parameters
    ----------
    p_drop : float
        Per-modality dropout probability (default: 0.1).
    """

    def __init__(self, p_drop: float = 0.1) -> None:
        super().__init__()
        self.p_drop = p_drop

    def forward(
        self,
        h_img: torch.Tensor,
        h_ehr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Possibly zero out one modality.

        Returns
        -------
        (h_img, h_ehr) — same shapes, possibly with one branch zeroed.
        """
        if not self.training:
            return h_img, h_ehr

        drop_img = torch.rand(1).item() < self.p_drop
        drop_ehr = torch.rand(1).item() < self.p_drop

        # Never drop both.
        if drop_img and drop_ehr:
            if torch.rand(1).item() < 0.5:
                drop_img = False
            else:
                drop_ehr = False

        if drop_img:
            h_img = torch.zeros_like(h_img)
        if drop_ehr:
            h_ehr = torch.zeros_like(h_ehr)

        return h_img, h_ehr


# ──────────────────────────────────────────────────────────────────────
# Full Multimodal Classifier
# ──────────────────────────────────────────────────────────────────────

class MultimodalClassifier(nn.Module):
    """End-to-end multimodal model: Image + EHR → pathology predictions.

    Implements the pseudo-algorithm::

        h_A ← Proj_A(E_A(x_A))              # image features
        h_B ← Proj_B(E_B(x_B))              # EHR features
        if mask_B == 0: h_B ← learned_absent_token
        z   ← F(h_A, h_B)                   # cross-attention fusion
        ŷ   ← H(z)                          # classification head
        L   ← L_task(ŷ, y) + λ · L_aux      # BCE + optional unimodal aux losses

    Parameters
    ----------
    d : int
        Shared embedding dimension.
    num_ehr_features : int
        Number of tabular EHR input features.
    num_classes : int
        Number of output pathology labels (default: 14).
    n_heads : int
        Number of attention heads in cross-attention and Transformer.
    modality_dropout : float
        Probability of dropping a modality during training.
    pretrained_backbone : bool
        Whether to load ImageNet-pretrained ResNet-50 weights.
    freeze_backbone : bool
        Whether to freeze ResNet-50 backbone weights.
    """

    def __init__(
        self,
        d: int = _DEFAULT_D,
        num_ehr_features: int = _DEFAULT_EHR_FEATURES,
        num_classes: int = _DEFAULT_N_CLASSES,
        n_heads: int = _DEFAULT_HEADS,
        modality_dropout: float = 0.1,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.d = d
        self.num_classes = num_classes

        # Encoders.
        self.image_encoder = ImageEncoder(
            d=d,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )
        self.ehr_encoder = EHREncoder(
            num_features=num_ehr_features, d=d, n_heads=n_heads
        )

        # Learned token for absent EHR modality.
        self.absent_ehr_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Fusion.
        self.modality_dropout = ModalityDropout(p_drop=modality_dropout)
        self.fusion = CrossAttentionFusion(d=d, n_heads=n_heads)

        # Classification head.
        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d, num_classes),
        )

        # Auxiliary unimodal heads (for auxiliary losses).
        self.aux_img_head = nn.Linear(d, num_classes)
        self.aux_ehr_head = nn.Linear(d, num_classes)

    def forward(
        self,
        x_img: torch.Tensor,
        x_ehr: torch.Tensor,
        mask_ehr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        x_img : ``(B, 1, H, W)``
            Grayscale chest X-ray images.
        x_ehr : ``(B, num_features)``
            Tabular EHR feature vector.
        mask_ehr : ``(B,)`` or None
            Binary mask: 1 = EHR available, 0 = absent.  When ``None`` all
            EHR data is assumed present.

        Returns
        -------
        dict with keys:
            ``logits``     — ``(B, num_classes)`` main prediction logits.
            ``aux_img``    — ``(B, num_classes)`` image-only auxiliary logits.
            ``aux_ehr``    — ``(B, num_classes)`` EHR-only auxiliary logits.
            ``h_img``      — ``(B, N, d)`` image features (for contrastive loss).
            ``h_ehr``      — ``(B, M, d)`` EHR features (for contrastive loss).
            ``fused``      — ``(B, d)`` fused representation.
        """
        # ── Encode each modality ──────────────────────────────────
        h_img = self.image_encoder(x_img)   # (B, N, d)
        h_ehr = self.ehr_encoder(x_ehr)     # (B, M, d)

        # ── Handle absent EHR modality ────────────────────────────
        if mask_ehr is not None:
            # Replace absent EHR tokens with learnable absent token.
            mask = mask_ehr.view(-1, 1, 1).float()  # (B, 1, 1)
            absent = self.absent_ehr_token.expand(
                h_ehr.size(0), h_ehr.size(1), -1
            )
            h_ehr = h_ehr * mask + absent * (1.0 - mask)

        # ── Modality dropout (training only) ──────────────────────
        h_img_drop, h_ehr_drop = self.modality_dropout(h_img, h_ehr)

        # ── Cross-attention fusion ────────────────────────────────
        fused = self.fusion(h_img_drop, h_ehr_drop)  # (B, d)

        # ── Classification head ───────────────────────────────────
        logits = self.classifier(fused)  # (B, num_classes)

        # ── Auxiliary unimodal predictions ────────────────────────
        aux_img = self.aux_img_head(h_img.mean(dim=1))  # (B, num_classes)
        aux_ehr = self.aux_ehr_head(h_ehr.mean(dim=1))  # (B, num_classes)

        return {
            "logits": logits,
            "aux_img": aux_img,
            "aux_ehr": aux_ehr,
            "h_img": h_img,
            "h_ehr": h_ehr,
            "fused": fused,
        }


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────

def test_forward_pass() -> None:
    """Create dummy data and verify all output dimensions.

    This function is a quick sanity check that can be run standalone::

        python -m src.fusion.modules
    """
    B = 4
    H = W = _DEFAULT_IMG_SIZE
    num_ehr = _DEFAULT_EHR_FEATURES
    num_classes = _DEFAULT_N_CLASSES

    model = MultimodalClassifier(
        d=_DEFAULT_D,
        num_ehr_features=num_ehr,
        num_classes=num_classes,
        pretrained_backbone=False,  # skip download in test
    )
    model.eval()

    x_img = torch.randn(B, 1, H, W)
    x_ehr = torch.randn(B, num_ehr)
    mask_ehr = torch.tensor([1, 1, 0, 1])  # patient 3 has no EHR

    with torch.no_grad():
        out = model(x_img, x_ehr, mask_ehr=mask_ehr)

    # Verify shapes.
    assert out["logits"].shape == (B, num_classes), (
        f"logits shape mismatch: {out['logits'].shape}"
    )
    assert out["aux_img"].shape == (B, num_classes), (
        f"aux_img shape mismatch: {out['aux_img'].shape}"
    )
    assert out["aux_ehr"].shape == (B, num_classes), (
        f"aux_ehr shape mismatch: {out['aux_ehr'].shape}"
    )
    assert out["fused"].shape == (B, _DEFAULT_D), (
        f"fused shape mismatch: {out['fused'].shape}"
    )
    assert out["h_img"].ndim == 3 and out["h_img"].shape[0] == B, (
        f"h_img shape unexpected: {out['h_img'].shape}"
    )
    assert out["h_ehr"].shape == (B, num_ehr, _DEFAULT_D), (
        f"h_ehr shape mismatch: {out['h_ehr'].shape}"
    )

    print("✓ All dimension checks passed.")
    print(f"  logits : {out['logits'].shape}")
    print(f"  aux_img: {out['aux_img'].shape}")
    print(f"  aux_ehr: {out['aux_ehr'].shape}")
    print(f"  fused  : {out['fused'].shape}")
    print(f"  h_img  : {out['h_img'].shape}")
    print(f"  h_ehr  : {out['h_ehr'].shape}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_forward_pass()
