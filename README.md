# 🔥🛡️⚔️ TorchAdv

TorchAdv is a Python package designed to facilitate the creation and execution of adversarial attacks on PyTorch models. This library aims to provide easy-to-use tools for generating adversarial examples, evaluating model robustness, and implementing state-of-the-art adversarial attack methods.

## Features

- **Adversarial Attacks**: Implementations of popular adversarial attacks such as FGSM, PGD, and more.
- **Compatibility**: Designed to work seamlessly with PyTorch models.
- **Customizable**: Easily extendable to include new attack methods or custom functionality.

## Installation

Install the package via pip:

```bash
pip install torchadv
```

## Usage

Here is a simple example of how to use TorchAdv to perform an FGSM attack on a PyTorch model:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

from torchadv.attacks import PGD

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Load an image and preprocess it
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image = Image.open('path_to_image.jpg')
orig = transform(image).unsqueeze(0)

# Define the target label
target_label = torch.tensor([your_target_label])

# Perform the attack
attack = PGD(model)
adv = attack(orig, target_label)
```

## Contributing

Contributions are welcome! If you have any ideas for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

