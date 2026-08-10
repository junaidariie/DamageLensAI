import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download


class Car_Classifier_Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(weights="DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_resnet_model_from_hf(
    repo_id: str,
    filename: str = "car-damage-classifier.pt",
    num_classes: int = 6,
    device: str = "cpu",
    hf_token: str = None,
) -> torch.nn.Module:
    """
    Downloads and loads the state_dict for Car_Classifier_Resnet from Hugging Face Hub.
    """
    print(f"Downloading checkpoint '{filename}' from Hugging Face repo '{repo_id}'...")
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=hf_token
    )

    model = Car_Classifier_Resnet(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ ResNet-18 model loaded successfully.")
    return model