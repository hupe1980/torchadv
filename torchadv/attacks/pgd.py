from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .attack import Attack
from .fgsm import FGSM
from ..utils import clip_tensor


@dataclass(kw_only=True)
class PGD(Attack):
    """
    Projected Gradient Descent (PGD) attack class.
    Inherits from Attack class.

    Papers:
        - https://arxiv.org/pdf/1607.02533.pdf
        - https://arxiv.org/pdf/1706.06083.pdf
    """

    eps: float = 8 / 255  # Maximum perturbation magnitude
    norm: float | int = np.inf  # Norm to use for perturbation calculation
    rand_init: bool = True  # Flag indicating whether to use random initialization
    rand_minmax: float | None = None  # Range for random initialization
    clip_min: float | None = None  # Minimum value for clipping adversarial examples
    clip_max: float | None = None  # Maximum value for clipping adversarial examples
    criterion: Callable = field(
        default_factory=lambda: torch.nn.CrossEntropyLoss()
    )  # Loss criterion for computing the adversarial loss
    alpha: float = 2 / 255  # Step size for each iteration of PGD
    steps: int = 40  # Number of iterations for PGD

    def __post_init__(self) -> None:
        """
        Initialize the PGD attack instance.
        """
        if not self.rand_minmax:
            # If rand_minmax is not provided, set it to eps
            self.rand_minmax = self.eps

        # Initialize FGSM attack instance for inner adversarial perturbation computation
        self._fgsm = FGSM(
            model=self.model,
            eps=self.alpha,
            norm=self.norm,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            criterion=self.criterion,
        )

    def _call_impl(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        """
        Implementation of the PGD attack.

        Args:
            inputs (torch.Tensor): The input tensor (original example).
            labels (torch.Tensor, optional): The target labels tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The adversarially perturbed input tensor (adversarial example).
        """
        if self.rand_init:
            # Randomly initialize perturbation within rand_minmax bounds
            eta = torch.zeros_like(inputs).uniform_(-self.rand_minmax, self.rand_minmax)  # type: ignore[operator, arg-type]
        else:
            eta = torch.zeros_like(inputs)

        # Clamp perturbation to ensure it stays within [-eps, eps]
        eta = torch.clamp(eta, min=-self.eps, max=self.eps)

        # Initial adversarial example
        adv = inputs + eta

        # Clip adversarial example to ensure it stays within specified bounds
        adv = clip_tensor(adv, min_val=self.clip_min, max_val=self.clip_max)

        for _ in range(self.steps):
            # Apply FGSM to compute adversarial perturbation
            adv = self._fgsm(inputs=adv, labels=labels)

            # Calculate perturbation from original example
            eta = adv - inputs
            # Clamp perturbation
            eta = torch.clamp(eta, min=-self.eps, max=self.eps)
            # Update adversarial example
            adv = inputs + eta

            # Perform the clipping again.
            # Although FGSM initially handled this, the subtraction and re-application of eta
            # can introduce minor numerical errors.
            adv = clip_tensor(adv, min_val=self.clip_min, max_val=self.clip_max)

        # Return the adversarially perturbed input tensor
        return adv
