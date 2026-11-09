"""Loss functions for multimodal chest X-ray + EHR classification.

Provides:
- :class:`MultimodalLoss` — combined task loss (BCEWithLogitsLoss for
  multi-label pathology classification), auxiliary unimodal losses, and an
  optional contrastive alignment term.
- :class:`ContrastiveAlignmentLoss` — InfoNCE-style loss that encourages
  image and EHR representations to be close in the shared embedding space
  for corresponding patients and far apart for non-corresponding ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Contrastive Alignment Loss
# ──────────────────────────────────────────────────────────────────────

class ContrastiveAlignmentLoss(nn.Module):
    """InfoNCE-style contrastive loss for cross-modal alignment.

    Aligns image and EHR representations in a shared embedding space so that
    paired (same-patient) embeddings are pulled together while unpaired
    embeddings are pushed apart.

    The loss is symmetric: we compute ``L_img→ehr`` (image anchors, EHR
    positives) and ``L_ehr→img`` (EHR anchors, image positives), then
    average.

    Parameters
    ----------
    temperature : float
        Softmax temperature τ.  Lower → sharper distribution (default 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_img: torch.Tensor,
        z_ehr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Parameters
        ----------
        z_img : ``(B, d)``
            L2-normalised image embeddings (one per sample).
        z_ehr : ``(B, d)``
            L2-normalised EHR embeddings (one per sample).

        Returns
        -------
        Scalar loss tensor.
        """
        # L2 normalise (in case caller forgot).
        z_img = F.normalize(z_img, dim=-1)
        z_ehr = F.normalize(z_ehr, dim=-1)

        # Cosine similarity matrix: (B, B).
        logits = z_img @ z_ehr.T / self.temperature

        # Labels: diagonal = positive pairs.
        B = z_img.size(0)
        labels = torch.arange(B, device=z_img.device)

        # Symmetric cross-entropy.
        loss_i2e = F.cross_entropy(logits, labels)
        loss_e2i = F.cross_entropy(logits.T, labels)

        return (loss_i2e + loss_e2i) / 2.0


# ──────────────────────────────────────────────────────────────────────
# Multimodal Combined Loss
# ──────────────────────────────────────────────────────────────────────

class MultimodalLoss(nn.Module):
    """Combined loss for multimodal pathology classification.

    .. math::

        L = L_{task}(\\hat{y}, y)
            + \\lambda_{aux} \\cdot \\bigl[ L_{aux\\_img} + L_{aux\\_ehr} \\bigr]
            + \\lambda_{con} \\cdot L_{contrastive}

    Components
    ----------
    * **Task loss** — ``BCEWithLogitsLoss`` on the main fused logits
      (multi-label classification of 14 pathology findings).
    * **Auxiliary unimodal losses** — ``BCEWithLogitsLoss`` on the auxiliary
      image-only and EHR-only heads.  These provide gradient signal to each
      encoder independently, improving single-modality performance and
      robustness to missing modalities.
    * **Contrastive alignment loss** — :class:`ContrastiveAlignmentLoss`
      encouraging paired image–EHR representations to align in the shared
      embedding space.

    Parameters
    ----------
    lambda_aux : float
        Weight for auxiliary unimodal losses (default 0.3).
    lambda_contrastive : float
        Weight for the contrastive alignment loss (default 0.1).
    use_contrastive : bool
        Whether to include the contrastive term (default ``True``).
    pos_weight : Tensor | None
        Per-class positive weight for ``BCEWithLogitsLoss`` to handle class
        imbalance.  Shape ``(num_classes,)``.
    temperature : float
        Temperature for contrastive loss (default 0.07).
    label_smoothing : float
        Label smoothing applied to BCE targets (default 0.0).
    """

    def __init__(
        self,
        lambda_aux: float = 0.3,
        lambda_contrastive: float = 0.1,
        use_contrastive: bool = True,
        pos_weight: torch.Tensor | None = None,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self.lambda_aux = lambda_aux
        self.lambda_contrastive = lambda_contrastive
        self.use_contrastive = use_contrastive
        self.label_smoothing = label_smoothing

        # Main and auxiliary BCE losses (shared parameterisation).
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_aux = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Contrastive loss (optional).
        self.contrastive = ContrastiveAlignmentLoss(temperature=temperature)

    def _smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to binary targets."""
        if self.label_smoothing > 0.0:
            return y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return y

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        model_output : dict
            Output dict from :class:`MultimodalClassifier.forward` containing
            at least ``logits``, ``aux_img``, ``aux_ehr``, ``h_img``,
            ``h_ehr``.
        y : ``(B, num_classes)``
            Ground-truth binary labels.

        Returns
        -------
        dict with keys:
            ``loss``            — total scalar loss (for ``backward()``).
            ``loss_task``       — main BCE loss.
            ``loss_aux_img``    — auxiliary image-only loss.
            ``loss_aux_ehr``    — auxiliary EHR-only loss.
            ``loss_contrastive``— contrastive alignment loss (0 if disabled).
        """
        y_smooth = self._smooth_labels(y.float())

        # ── Task loss ─────────────────────────────────────────────
        loss_task = self.bce(model_output["logits"], y_smooth)

        # ── Auxiliary unimodal losses ─────────────────────────────
        loss_aux_img = self.bce_aux(model_output["aux_img"], y_smooth)
        loss_aux_ehr = self.bce_aux(model_output["aux_ehr"], y_smooth)

        # ── Contrastive alignment loss ────────────────────────────
        loss_contrastive = torch.tensor(0.0, device=y.device)
        if self.use_contrastive:
            # Pool spatial/sequential tokens to single vectors for contrastive.
            z_img = model_output["h_img"].mean(dim=1)  # (B, d)
            z_ehr = model_output["h_ehr"].mean(dim=1)  # (B, d)
            loss_contrastive = self.contrastive(z_img, z_ehr)

        # ── Total loss ────────────────────────────────────────────
        total = (
            loss_task
            + self.lambda_aux * (loss_aux_img + loss_aux_ehr)
            + self.lambda_contrastive * loss_contrastive
        )

        return {
            "loss": total,
            "loss_task": loss_task.detach(),
            "loss_aux_img": loss_aux_img.detach(),
            "loss_aux_ehr": loss_aux_ehr.detach(),
            "loss_contrastive": loss_contrastive.detach(),
        }
