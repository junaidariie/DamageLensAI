import logging
import torch
import torch.nn as nn
from torchvision import models
from transformers import ConvNextModel

logger = logging.getLogger(__name__)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        convnext_model_name="facebook/convnext-small-224"
    ):
        super().__init__()

        logger.info("Initializing Fusion model...")

        # EfficientNet-V2-S
        eff = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )

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

        # ConvNeXt
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

        fused_dim = self.eff_out_dim + self.cnx_out_dim

        self.fusion_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        logger.info("Fusion model initialized successfully.")

    def forward(self, pixel_values_eff, pixel_values_cnx):
        x_eff = self.eff_features(pixel_values_eff)
        x_eff = self.eff_avgpool(x_eff)
        x_eff = torch.flatten(x_eff, 1)

        cnx_out = self.cnx_backbone(
            pixel_values=pixel_values_cnx,
            return_dict=True
        )

        x_cnx = cnx_out.pooler_output

        fused = torch.cat([x_eff, x_cnx], dim=1)

        logits = self.fusion_head(fused)

        return logits


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    model = FusionClassifier(num_classes=6)

    eff_dummy = torch.randn(2, 3, 260, 260)
    cnx_dummy = torch.randn(2, 3, 224, 224)

    output = model(eff_dummy, cnx_dummy)

    print("Fusion output shape:", output.shape)