# utils/model/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri

class DomainClassifier(nn.Module):
    """
    A simple CNN to classify MRI data into domains (e.g., knee vs brain).
    [FIX] BatchNorm2d를 InstanceNorm2d로 교체하여 학습/검증 불일치 문제 해결.
    """
    def __init__(self, in_chans: int = 1, num_classes: int = 2, base_chans: int = 16):
        """
        Args:
            in_chans: Input channels. For RSS image, this is always 1.
            num_classes: Number of domains to classify.
            base_chans: Base number of channels for the first convolution layer.
        """
        super().__init__()

        # [FIX] BatchNorm2d를 InstanceNorm2d로 변경.
        # InstanceNorm은 채널 단위로 각 데이터에 대해 독립적으로 정규화를 수행합니다.
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, base_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_chans), # BatchNorm2d -> InstanceNorm2d
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(base_chans, base_chans * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_chans * 2), # BatchNorm2d -> InstanceNorm2d
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear((base_chans * 2) * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    # [FIX] self.norm 함수는 더 이상 필요 없으므로 제거합니다.
    # def norm(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kspace: Input k-space data of shape (B, C, H, W, 2)
        """
        # 1. k-space -> RSS image
        image_coils_complex = fastmri.ifft2c(kspace)
        rss_image = fastmri.rss_complex(image_coils_complex, dim=1)
        rss_image = rss_image.unsqueeze(1)
        
        # 2. [FIX] CNN Forward Pass
        # self.norm() 호출을 제거하고 Sequential 모델에 바로 입력합니다.
        logits = self.layers(rss_image)

        return logits
