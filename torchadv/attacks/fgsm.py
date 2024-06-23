from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..attack import Attack
from ..utils import clip_tensor, normalize_gradients


@dataclass(kw_only=True)
class FGSM(Attack):
    """
    Fast Gradient Sign Method (FGSM) attack class.
    Inherits from Attack class.

    Papers:
        - https://arxiv.org/abs/1412.6572
    """

    eps: float = 8 / 255  # Maximum perturbation magnitude
    norm: float | int = np.inf  # Norm to use for perturbation calculation
    clip_min: float | None = None  # Minimum value for clipping adversarial examples
    clip_max: float | None = None  # Maximum value for clipping adversarial examples
    criterion: Callable = field(
        default_factory=lambda: torch.nn.CrossEntropyLoss()
    )  # Loss criterion for computing the adversarial loss

    def _call_impl(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        """
        Implementation of the FGSM attack.

        Args:
            inputs (torch.Tensor): The input tensor (original example).
            labels (torch.Tensor, optional): The target labels tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The adversarially perturbed input tensor (adversarial example).
        """
        inputs = inputs.clone().detach().requires_grad_(True)

        is_targeted = labels is not None
        if not is_targeted:
            _, labels = torch.max(self.get_logits(inputs), 1)

        loss = self.criterion(self.get_logits(inputs), labels)

        if is_targeted:
            # Invert loss for targeted attack
            loss = -loss

        # Compute gradient of loss w.r.t. inputs tensor
        grad = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)[0]

        # Normalize gradient and scale by epsilon to get perturbation
        optimal_perturbation = normalize_gradients(grad, self.norm) * self.eps

        # Add perturbation to original example to obtain adversarial example
        adv = inputs + optimal_perturbation

        # Clip adversarial example to ensure it stays within specified bounds
        adv = clip_tensor(adv, min_val=self.clip_min, max_val=self.clip_max)

        # Return the adversarially perturbed inputs tensor
        return adv.detach()
