import pytest
import torch

from torchadv.utils import clip_tensor


@pytest.fixture
def input_tensor():
    return torch.tensor([-1.0, 0.5, 2.0, 3.0, 4.0])

def test_clip_tensor_both_limits(input_tensor):
    clipped_tensor = clip_tensor(input_tensor, min_val=-0.5, max_val=2.5)
    assert torch.allclose(clipped_tensor, torch.tensor([-0.5, 0.5, 2.0, 2.5, 2.5]))

def test_clip_tensor_min_limit(input_tensor):
    clipped_tensor = clip_tensor(input_tensor, min_val=0.0)
    assert torch.allclose(clipped_tensor, torch.tensor([0.0, 0.5, 2.0, 3.0, 4.0]))

def test_clip_tensor_max_limit(input_tensor):
    clipped_tensor = clip_tensor(input_tensor, max_val=2.0)
    assert torch.allclose(clipped_tensor, torch.tensor([-1.0, 0.5, 2.0, 2.0, 2.0]))

def test_clip_tensor_no_limits(input_tensor):
    clipped_tensor = clip_tensor(input_tensor)
    assert torch.allclose(clipped_tensor, input_tensor)

def test_clip_tensor_already_within_range(input_tensor):
    clipped_tensor = clip_tensor(input_tensor, min_val=-1.5, max_val=4.5)
    assert torch.allclose(clipped_tensor, input_tensor)
