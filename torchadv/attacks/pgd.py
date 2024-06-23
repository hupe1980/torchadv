from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .fgsm import FGSM
from ..attack import Attack
from ..utils import clip_tensor


@dataclass(kw_only=True)
class PGD(Attack):
    eps: float = 0.3
    norm: float | int = np.inf
    rand_init: bool = True
    rand_minmax: float | None = None
    clip_min: float | None = None
    clip_max: float | None = None
    criterion: Callable = field(default_factory=lambda: torch.nn.CrossEntropyLoss())
    eps_iter: float = 0.01
    steps: int = 40

    def __post_init__(self) -> None:
        if not self.rand_minmax:
            self.rand_minmax = self.eps

        self._fgsm = FGSM(
            model=self.model,
            eps=self.eps_iter,
            norm=self.norm,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            criterion=self.criterion,
        )

    def _call_impl(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        if self.rand_init:
            eta = torch.zeros_like(inputs).uniform_(-self.rand_minmax, self.rand_minmax)  # type: ignore[operator, arg-type]
        else:
            eta = torch.zeros_like(inputs)

        eta = torch.clamp(eta, min=-self.eps, max=self.eps)

        adv_x = inputs + eta

        adv_x = clip_tensor(adv_x, min_val=self.clip_min, max_val=self.clip_max)

        for _ in range(self.steps):
            adv_x = self._fgsm(inputs=adv_x, labels=labels)

            eta = adv_x - inputs
            eta = torch.clamp(eta, min=-self.eps, max=self.eps)
            adv_x = inputs + eta

            # Perform the clipping again.
            # Although FGSM initially handled this, the subtraction and re-application of eta
            # can introduce minor numerical errors.
            adv_x = clip_tensor(adv_x, min_val=self.clip_min, max_val=self.clip_max)

        return adv_x
