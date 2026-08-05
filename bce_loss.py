import torch
import torch.nn as nn


def two_way_margin(logits):
    """Convert DDP's [negative, positive] path logits to one binary logit."""

    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError(
            "DDP logits must have shape [batch, 2, classes]; "
            f"got {tuple(logits.shape)}"
        )
    return logits[:, 1, :] - logits[:, 0, :]


class BCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(BCELoss, self).__init__()

        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets
        """

        # Calculating Probabilities
        x_softmax = self.softmax(x)
        xs_pos = x_softmax[:, 1, :]

        xs_neg = x_softmax[:, 0, :]
        y = y.reshape(-1)
        xs_pos = xs_pos.reshape(-1)
        xs_neg = xs_neg.reshape(-1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()


class TwoWayAsymmetricLoss(nn.Module):
    """ASL applied to DDP's positive-minus-negative two-way logit margin.

    The default ``sum`` reduction intentionally matches :class:`BCELoss`.
    DDP therefore keeps its original outer ``loss_w`` and optimizer settings;
    only the per-label classification objective changes.
    """

    def __init__(
        self,
        gamma_neg=9.8,
        gamma_pos=0.0,
        clip=0.05,
        eps=1e-8,
        reduction="sum",
        detach_focal_weight=True,
    ):
        super().__init__()
        if gamma_neg < 0 or gamma_pos < 0:
            raise ValueError("gamma_neg and gamma_pos must be non-negative")
        if clip < 0 or clip >= 1:
            raise ValueError("clip must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if reduction not in ("sum", "mean", "none"):
            raise ValueError("reduction must be 'sum', 'mean', or 'none'")
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction
        self.detach_focal_weight = bool(detach_focal_weight)

    def elementwise_loss(self, logits, targets):
        margin = two_way_margin(logits).float()
        targets = targets.float()
        if margin.shape != targets.shape:
            raise ValueError(
                "DDP margin and targets must have equal [batch, classes] "
                f"shapes; got {tuple(margin.shape)} and {tuple(targets.shape)}"
            )

        positive_probability = torch.sigmoid(margin)
        negative_probability = 1.0 - positive_probability
        if self.clip > 0:
            negative_probability = (
                negative_probability + self.clip
            ).clamp(max=1.0)

        positive_loss = targets * torch.log(
            positive_probability.clamp_min(self.eps)
        )
        negative_targets = 1.0 - targets
        negative_loss = negative_targets * torch.log(
            negative_probability.clamp_min(self.eps)
        )

        positive_weight = (1.0 - positive_probability).pow(self.gamma_pos)
        negative_weight = (1.0 - negative_probability).pow(self.gamma_neg)
        focal_weight = (
            targets * positive_weight + negative_targets * negative_weight
        )
        if self.detach_focal_weight:
            focal_weight = focal_weight.detach()
        return -(positive_loss + negative_loss) * focal_weight

    def forward(self, logits, targets):
        element_loss = self.elementwise_loss(logits, targets)
        if not torch.isfinite(element_loss).all():
            raise FloatingPointError("DDP ASL produced a non-finite loss")
        if self.reduction == "sum":
            return element_loss.sum()
        if self.reduction == "mean":
            return element_loss.mean()
        return element_loss

    @torch.no_grad()
    def diagnostics(self, logits, targets):
        margin = two_way_margin(logits).float()
        targets = targets.float()
        probabilities = torch.sigmoid(margin)
        positive = targets.gt(0.5)
        negative = ~positive
        element_loss = self.elementwise_loss(logits, targets)

        def masked_mean(values, mask):
            if not mask.any():
                return 0.0
            return float(values[mask].mean().item())

        easy_negative = negative & probabilities.le(self.clip)
        negative_count = int(negative.sum().item())
        return {
            "margin_min": float(margin.min().item()),
            "margin_max": float(margin.max().item()),
            "margin_mean": float(margin.mean().item()),
            "positive_probability_mean": masked_mean(probabilities, positive),
            "negative_probability_mean": masked_mean(probabilities, negative),
            "positive_element_loss_mean": masked_mean(element_loss, positive),
            "negative_element_loss_mean": masked_mean(element_loss, negative),
            "positive_labels": int(positive.sum().item()),
            "negative_labels": negative_count,
            "easy_negative_labels": int(easy_negative.sum().item()),
            "easy_negative_fraction": (
                float(easy_negative.sum().item()) / negative_count
                if negative_count
                else 0.0
            ),
        }


def build_ddp_classification_loss(
    name,
    gamma_neg=9.8,
    gamma_pos=0.0,
    clip=0.05,
    eps=1e-8,
):
    if name == "two_way_bce":
        return BCELoss()
    if name == "asl":
        return TwoWayAsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            eps=eps,
            reduction="sum",
        )
    raise ValueError(f"Unsupported DDP classification loss: {name}")


@torch.no_grad()
def two_way_prediction_diagnostics(logits, targets, easy_negative_cutoff=0.05):
    """Loss-independent first-batch diagnostics for BCE/ASL comparison."""

    margin = two_way_margin(logits).float()
    targets = targets.float()
    if margin.shape != targets.shape:
        raise ValueError(
            f"Margin/target shape mismatch: {margin.shape} versus {targets.shape}"
        )
    probabilities = torch.sigmoid(margin)
    positive = targets.gt(0.5)
    negative = ~positive
    easy_negative = negative & probabilities.le(float(easy_negative_cutoff))

    def masked_mean(values, mask):
        return float(values[mask].mean().item()) if mask.any() else 0.0

    return {
        "margin_min": float(margin.min().item()),
        "margin_max": float(margin.max().item()),
        "margin_mean": float(margin.mean().item()),
        "positive_probability_mean": masked_mean(probabilities, positive),
        "negative_probability_mean": masked_mean(probabilities, negative),
        "positive_labels": int(positive.sum().item()),
        "negative_labels": int(negative.sum().item()),
        "easy_negative_labels": int(easy_negative.sum().item()),
        "easy_negative_fraction": (
            float(easy_negative.sum().item()) / int(negative.sum().item())
            if negative.any()
            else 0.0
        ),
    }
