"""EMOT-Net host with the source-faithful CCIM backdoor intervention."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..emot_net_ft.model import EMOTNetFTModel


CCIM_UPSTREAM_REPOSITORY = "https://github.com/ydk122024/CCIM"
CCIM_UPSTREAM_COMMIT = "d6a651f91d1c1c91faca862ddeea915df9314919"
CCIM_SOURCE_SHA256 = "2e1b2f1178fcdc556aee7efed743c624020d43c34ec2272f7415dee4ce776d26"


def _batch_norm_allow_singleton(norm: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
    """Keep source BatchNorm while retaining a singleton final batch."""

    if norm.training and values.shape[0] == 1:
        return F.batch_norm(
            values,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            eps=norm.eps,
        )
    return norm(values)


class CCIMResidualClassifier(nn.Module):
    """The exact 128-D residual projection in the official ``CCIM.py``."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def gelu(values: torch.Tensor) -> torch.Tensor:
        # Preserve the tanh approximation implemented explicitly upstream.
        return 0.5 * values * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (values + 0.044715 * torch.pow(values, 3))
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = _batch_norm_allow_singleton(self.norm, values)
        values = self.dropout(self.gelu(self.fc1(values)))
        values = self.dropout(self.fc2(values))
        return residual + 0.3 * values


class CCIMBackdoorIntervention(nn.Module):
    """CCIM's EMOTIC dot-product or additive backdoor adjustment."""

    def __init__(
        self,
        joint_dim: int,
        confounder_dim: int,
        hidden_dim: int = 128,
        attention_dim: int = 256,
        strategy: str = "dp_cause",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if strategy not in {"dp_cause", "ad_cause"}:
            raise ValueError("CCIM strategy must be dp_cause or ad_cause")
        if min(joint_dim, confounder_dim, hidden_dim, attention_dim) <= 0:
            raise ValueError("CCIM dimensions must be positive")
        self.joint_dim = int(joint_dim)
        self.confounder_dim = int(confounder_dim)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.strategy = strategy

        self.w_h = nn.Parameter(torch.Tensor(self.joint_dim, self.hidden_dim))
        self.w_g = nn.Parameter(torch.Tensor(self.confounder_dim, self.hidden_dim))
        self.query = nn.Linear(self.joint_dim, self.attention_dim, bias=False)
        self.key = nn.Linear(self.confounder_dim, self.attention_dim, bias=False)
        self.w_t = (
            nn.Linear(self.attention_dim, 1, bias=False)
            if strategy == "ad_cause"
            else None
        )
        self.classifier = CCIMResidualClassifier(self.hidden_dim, dropout=dropout)
        nn.init.xavier_normal_(self.w_h)
        nn.init.xavier_normal_(self.w_g)

    @staticmethod
    def _validate_dictionary(
        dictionary: torch.Tensor,
        prior: torch.Tensor,
        confounder_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dictionary.ndim != 2 or dictionary.shape[1] != confounder_dim:
            raise ValueError("CCIM dictionary shape differs from confounder_dim")
        if prior.ndim == 1:
            prior = prior.unsqueeze(1)
        if prior.shape != (dictionary.shape[0], 1):
            raise ValueError("CCIM prior must be [dictionary_size, 1]")
        if not torch.isfinite(dictionary).all() or not torch.isfinite(prior).all():
            raise ValueError("CCIM dictionary and prior must be finite")
        if (prior < 0).any() or not torch.isclose(
            prior.sum(), prior.new_tensor(1.0), atol=1.0e-6, rtol=1.0e-6
        ):
            raise ValueError("CCIM prior must be non-negative and sum to one")
        return dictionary, prior

    def intervene(
        self,
        joint_features: torch.Tensor,
        dictionary: torch.Tensor,
        prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dictionary, prior = self._validate_dictionary(
            dictionary, prior, self.confounder_dim
        )
        if joint_features.ndim != 2 or joint_features.shape[1] != self.joint_dim:
            raise ValueError("CCIM joint feature width differs from joint_dim")
        if self.strategy == "dp_cause":
            # Upstream deliberately scales by sqrt(context feature width), not
            # sqrt(attention width); keep this seemingly unusual choice exact.
            affinity = torch.matmul(
                self.query(joint_features), self.key(dictionary).transpose(0, 1)
            ) / math.sqrt(float(self.confounder_dim))
        else:
            fused = torch.tanh(
                self.query(joint_features).unsqueeze(1)
                + self.key(dictionary).unsqueeze(0)
            )
            if self.w_t is None:
                raise RuntimeError("CCIM additive scorer is missing")
            affinity = self.w_t(fused).squeeze(2)
        attention = torch.softmax(affinity, dim=1)
        # Keep the official broadcasting expression exact. ``dictionary`` and
        # ``prior`` are [K,D] and [K,1], while the attention is [N,K,1].
        intervention = (
            attention.unsqueeze(2) * dictionary * prior
        ).sum(dim=1)
        return intervention, attention

    def forward(
        self,
        joint_features: torch.Tensor,
        dictionary: torch.Tensor,
        prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        intervention, attention = self.intervene(
            joint_features, dictionary, prior
        )
        hidden = torch.matmul(joint_features, self.w_h) + torch.matmul(
            intervention, self.w_g
        )
        return self.classifier(hidden), attention


class EMOTNetCCIMFTModel(EMOTNetFTModel):
    """EMOT-Net joint feature followed by CCIM and expanding FT heads."""

    def __init__(
        self,
        confounder_dictionary: torch.Tensor,
        confounder_prior: torch.Tensor,
        *,
        context_encoder: Optional[nn.Module] = None,
        body_encoder: Optional[nn.Module] = None,
        fusion_dim: int = 256,
        ccim_hidden_dim: int = 128,
        ccim_attention_dim: int = 256,
        ccim_strategy: str = "dp_cause",
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            context_encoder=context_encoder,
            body_encoder=body_encoder,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )
        dictionary = torch.as_tensor(confounder_dictionary).detach().float().clone()
        prior = torch.as_tensor(confounder_prior).detach().float().clone()
        if prior.ndim == 1:
            prior = prior.unsqueeze(1)
        if dictionary.ndim != 2:
            raise ValueError("CCIM confounder dictionary must be two-dimensional")
        self.ccim_hidden_dim = int(ccim_hidden_dim)
        self.ccim = CCIMBackdoorIntervention(
            joint_dim=self.fusion_dim,
            confounder_dim=int(dictionary.shape[1]),
            hidden_dim=self.ccim_hidden_dim,
            attention_dim=int(ccim_attention_dim),
            strategy=ccim_strategy,
            dropout=dropout,
        )
        dictionary, prior = self.ccim._validate_dictionary(
            dictionary, prior, self.ccim.confounder_dim
        )
        self.register_buffer("confounder_dictionary", dictionary, persistent=True)
        self.register_buffer("confounder_prior", prior, persistent=True)
        self.last_attention: Optional[torch.Tensor] = None

    @property
    def dictionary_size(self) -> int:
        return int(self.confounder_dictionary.shape[0])

    @property
    def confounder_dim(self) -> int:
        return int(self.confounder_dictionary.shape[1])

    def add_head(self, classes: int) -> nn.Linear:
        if classes <= 0:
            raise ValueError("A task head must contain at least one class")
        head = nn.Linear(self.ccim_hidden_dim, int(classes))
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        self.heads.append(head)
        self._head_sizes.append(int(classes))
        return head

    def restore_heads(self, sizes: Iterable[int]) -> None:
        if self.heads:
            raise RuntimeError("restore_heads requires an empty model")
        for size in sizes:
            self.add_head(int(size))

    def encode_joint(self, images: torch.Tensor) -> torch.Tensor:
        return super().encode(images)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features, attention = self.ccim(
            self.encode_joint(images),
            self.confounder_dictionary,
            self.confounder_prior,
        )
        self.last_attention = attention.detach()
        return features

    def dictionary_metadata(self) -> Mapping[str, object]:
        return {
            "dictionary_size": self.dictionary_size,
            "confounder_dim": self.confounder_dim,
            "prior_sum": float(self.confounder_prior.sum().detach().cpu()),
            "strategy": self.ccim.strategy,
        }
