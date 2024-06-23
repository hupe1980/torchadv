import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def normalize_gradients(grad: torch.Tensor, norm: float | int = np.inf) -> torch.Tensor: 
    if norm == np.inf:
        return grad.sign()
    
    elif norm == 1 or norm == 2:
        # Get the batch size
        batch_size = grad.size(0)
        
        # Compute the norm based on norm value (1 for L1, 2 for L2)
        norm_type = norm
        
        # Compute the norm of the gradients
        grad_norms = (
            torch.norm(grad.view(batch_size, -1), p=norm_type, dim=1)
            + 1e-12  # To avoid division by zero
        )

        # Normalize the gradients by their norm
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)

        return grad
        
    else:
        raise NotImplementedError(
            "Only Linf, L1 and L2 norms are currently implemented."
        )
    
def get_available_device() -> torch.device:
    """
    Returns the available device for PyTorch computations.

    If a GPU is available, it returns 'cuda'.
    If MPS is available, it returns 'mps'.
    Otherwise, it returns 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device

def get_model_device(model: nn.Module) -> torch.device:
    """
    Returns the device on which the model's parameters are located.

    Args:
        model (torch.nn.Module): The model for which the device is to be determined.

    Returns:
        torch.device: The device on which the model's parameters are located.
    """
    return next(model.parameters()).device


def show_images_diff(original_img: np.ndarray, original_label: str, adversarial_img: np.ndarray, adversarial_label: str) -> None:
    """
    Displays the original, adversarial, and difference images along with their labels.
    
    Args:
        original_img (np.ndarray): The original image.
        original_label (str): The label of the original image.
        adversarial_img (np.ndarray): The adversarial image.
        adversarial_label (str): The label of the adversarial image.
    """
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display original image
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original [={original_label}]')
    axes[0].axis('off')
    
    # Display adversarial image
    axes[1].imshow(adversarial_img)
    axes[1].set_title(f'Adversarial [={adversarial_label}]')
    axes[1].axis('off')
    
    # Compute and display the difference image
    difference = adversarial_img - original_img
    
    
    # Normalize the difference for display purposes
    difference_normalized = (difference / np.abs(difference).max()) / 2.0 + 0.5
    axes[2].imshow(difference_normalized, cmap='gray')
    axes[2].set_title('Adversarial-Original')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    plt.show()

    # Compute the L0 norm: number of non-zero elements in the difference
    l0 = np.count_nonzero(difference)
    # Compute the L2 norm: Euclidean distance between the original and adversarial images
    l2 = np.linalg.norm(difference)
    
    print(f"L0={l0} L2={l2:.4f}")

def clip_tensor(tensor: torch.Tensor, min_val: float | None=None, max_val:float | None=None):
    """
    Perform both-sided clipping on the input tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor to be clipped.
        min_val (float, optional): The minimum value to clip to. Default is None.
        max_val (float, optional): The maximum value to clip to. Default is None.
    
    Returns:
        torch.Tensor: The clipped tensor.
    """
    if min_val is not None and max_val is not None:
        tensor = torch.clamp(tensor, min=min_val, max=max_val)
    elif min_val is not None:
        tensor = torch.clamp(tensor, min=min_val)
    elif max_val is not None:
        tensor = torch.clamp(tensor, max=max_val)
    
    return tensor

