import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ConvNextModel
from huggingface_hub import hf_hub_download


# ==========================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================
class FusionClassifier(nn.Module):
    def __init__(self, num_classes, convnext_model_name="facebook/convnext-small-224"):
        super().__init__()

        # EfficientNet-V2-S backbone
        eff = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        for param in eff.parameters():
            param.requires_grad = False

        for param in eff.features[5].parameters():
            param.requires_grad = True
        for param in eff.features[6].parameters():
            param.requires_grad = True
        for param in eff.features[7].parameters():
            param.requires_grad = True

        self.eff_features = eff.features
        self.eff_avgpool = eff.avgpool
        self.eff_out_dim = eff.classifier[1].in_features

        # ConvNeXt-Small backbone
        cnx = ConvNextModel.from_pretrained(convnext_model_name)

        for param in cnx.parameters():
            param.requires_grad = False

        for param in cnx.encoder.stages[2].parameters():
            param.requires_grad = True
        for param in cnx.encoder.stages[3].parameters():
            param.requires_grad = True
        for param in cnx.layernorm.parameters():
            param.requires_grad = True

        self.cnx_backbone = cnx
        self.cnx_out_dim = 768

        # Fusion head
        fused_dim = self.eff_out_dim + self.cnx_out_dim
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values_eff, pixel_values_cnx):
        x_eff = self.eff_features(pixel_values_eff)
        x_eff = self.eff_avgpool(x_eff)
        x_eff = torch.flatten(x_eff, 1)

        cnx_out = self.cnx_backbone(pixel_values=pixel_values_cnx, return_dict=True)
        x_cnx = cnx_out.pooler_output

        fused = torch.cat([x_eff, x_cnx], dim=1)
        logits = self.fusion_head(fused)

        return logits


# ==========================================
# 2. HUGGING FACE DOWNLOAD & LOAD FUNCTION
# ==========================================
def load_fusion_model_from_hf(
    repo_id: str = "junaid17/best_fusion_model_fp16",
    filename: str = "best_fusion_model_fp16.pt",
    num_classes: int = 10,  # Replace with your actual number of classes
    device: str = "cpu",
) -> nn.Module:
    """Downloads weights from Hugging Face Hub and loads into FusionClassifier."""
    print(f"Downloading checkpoint from Hugging Face Hub: '{repo_id}/{filename}'...")
    
    # Download weights file from Hugging Face Hub
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # Instantiate Model
    model = FusionClassifier(num_classes=num_classes)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    # Ensure model is float32 for stable ONNX export
    model = model.float().to(device)
    model.eval()
    
    print("✅ Model loaded successfully from Hugging Face.")
    return model