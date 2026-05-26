import logging
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        logger.info("Initializing ResNet18 model...")

        self.model = models.resnet18(weights="DEFAULT")

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last layers
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Custom classifier head
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        logger.info("ResNet18 model initialized successfully.")

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    model = CarClassifierResNet(num_classes=6)

    dummy_input = torch.randn(2, 3, 128, 128)

    output = model(dummy_input)

    print("Output shape:", output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)