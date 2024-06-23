from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..attack import Attack
from ..utils import clip_tensor, normalize_gradients


@dataclass(kw_only=True)
class FGSM(Attack):
    eps: float = 0.3
    norm: float | int = np.inf
    clip_min: float | None = None
    clip_max: float | None = None
    criterion: Callable = field(default_factory=lambda: torch.nn.CrossEntropyLoss())

    def _call_impl(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        inputs = inputs.clone().detach().requires_grad_(True)

        is_targeted = labels is not None
        if not is_targeted:
            _, labels = torch.max(self.get_logits(inputs), 1)

        loss = self.criterion(self.get_logits(inputs), labels)

        if is_targeted:
            loss = -loss

        grad = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)[0]

        optimal_perturbation = normalize_gradients(grad, self.norm) * self.eps

        # Add perturbation to original example to obtain adversarial example
        adv_x = inputs + optimal_perturbation

        adv_x = clip_tensor(adv_x, min_val=self.clip_min, max_val=self.clip_max)

        return adv_x
