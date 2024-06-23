from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from .utils import get_model_device


@dataclass()
class Attack(ABC):
    model: nn.Module

    def __post_init__(self) -> None:
        self._device = get_model_device(self.model)

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        return self._call_impl(inputs=inputs, labels=labels, **kwargs)

    @abstractmethod
    def _call_impl(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        raise NotImplementedError

    def get_logits(self, inputs: torch.Tensor, **kwargs):
        return self.model(inputs, **kwargs)
