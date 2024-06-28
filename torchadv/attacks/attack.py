from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..utils import get_model_device


@dataclass
class Attack(ABC):
    """
    Abstract base class representing an attack against a neural network model.

    Attributes:
        model (nn.Module): The neural network model to be attacked.
    """

    model: nn.Module

    def __post_init__(self) -> None:
        """Initialize the Attack instance by setting the device the model is on."""
        self._device = get_model_device(self.model)

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Apply the attack to the given inputs.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor, optional): The target labels tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The adversarially perturbed inputs.
        """
        return self._call_impl(inputs=inputs, labels=labels, **kwargs)

    @abstractmethod
    def _call_impl(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses to apply the attack.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor, optional): The target labels tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The adversarially perturbed inputs.
        """
        raise NotImplementedError

    def get_logits(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get logits from the model for the given inputs.

        Args:
            inputs (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The logits tensor.
        """
        return self.model(inputs, **kwargs)

    def predict_labels(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predict labels from the given inputs using the model's logits.

        Args:
            inputs (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The predicted labels tensor.
        """
        _, labels = torch.max(self.get_logits(inputs, **kwargs), dim=1)
        return labels
