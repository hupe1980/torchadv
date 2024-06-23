from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from .utils import get_model_device


@dataclass
class Attack(ABC):
    """
    Abstract base class representing an attack against a neural network model.
    """

    model: nn.Module

    def __post_init__(self) -> None:
        """
        Initialize the Attack instance.
        """
        self._device = get_model_device(self.model)

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        """
        Call method to apply the attack.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor, optional): The target labels tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The adversarially perturbed inputs.
        """
        return self._call_impl(inputs=inputs, labels=labels, **kwargs)

    @abstractmethod
    def _call_impl(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        """
        Abstract method to be implemented by subclasses.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor, optional): The target labels tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The adversarially perturbed inputs.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    def get_logits(self, inputs: torch.Tensor, **kwargs):
        """
        Get logits from the model for given inputs.

        Args:
            inputs (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The logits tensor.
        """
        return self.model(inputs, **kwargs)
