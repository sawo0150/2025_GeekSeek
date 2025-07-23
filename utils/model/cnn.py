# utils/model/cnn.py
import torch
import torch.nn as nn
import fastmri

class DomainClassifier(nn.Module):
    """
    A simple CNN to classify MRI data into domains (e.g., knee_x4, brain_x8).
    It takes k-space data as input, converts it to image space, and classifies.
    """
    def __init__(self, in_chans: int = 2, num_classes: int = 4, base_chans: int = 16):
        """
        Args:
            in_chans: Input channels. 2 for complex (real, imag).
            num_classes: Number of domains to classify.
            base_chans: Base number of channels for the first conv layer.
        """
        super().__init__()
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, base_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_chans),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(base_chans, base_chans * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_chans * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(base_chans * 2, base_chans * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_chans * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_chans * 4, base_chans * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_chans * 2, num_classes)
        )

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """ Instance normalization """
        # x shape: (B, C, H, W)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-6)

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kspace: Input k-space data of shape (B, C, H, W, 2)
        """
        # 1. k-space to image space
        # (B, C, H, W, 2) -> (B, C, H, W) complex
        x_complex = torch.view_as_complex(kspace)
        # IFFT -> (B, C, H, W) complex
        x_img_space = fastmri.ifft2c(x_complex)
        
        # 2. Reshape for CNN: treat coils and real/imag as channels
        # (B, C, H, W) complex -> (B, C, H, W, 2) real
        x_img_space_real = torch.view_as_real(x_img_space)
        # (B, C, H, W, 2) -> (B, C*2, H, W)
        b, c, h, w, _ = x_img_space_real.shape
        x = x_img_space_real.permute(0, 1, 4, 2, 3).reshape(b, c * 2, h, w)

        # 3. Normalize and pass through CNN
        x = self.norm(x)
        logits = self.layers(x)

        return logits
