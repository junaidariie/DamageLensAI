import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from transformers import ConvNextModel, ConvNextImageProcessor

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- RESNET MODEL ----------------
class Car_Classifier_Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        logger.info("Initializing ResNet18 architecture...")

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

        logger.info("ResNet architecture initialized successfully.")

    def forward(self, x):
        return self.model(x)


class ResnetCarDamagePredictor:
    def __init__(self, checkpoint_path, class_map):
        logger.info("Initializing ResNet predictor...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map

        logger.info(f"Using device for ResNet: {self.device}")

        self.test_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        try:
            self.model = Car_Classifier_Resnet(num_classes=len(class_map))

            logger.info(f"Loading ResNet checkpoint from: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            logger.info("ResNet model loaded successfully.")

        except Exception as e:
            logger.exception("Failed to load ResNet model.")
            raise RuntimeError(f"Failed to load ResNet model: {str(e)}")

    def resnet_predict(self, image_input):
        logger.info("Starting ResNet prediction...")

        try:
            if isinstance(image_input, str):
                logger.info(f"Loading image from file path: {image_input}")
                image = Image.open(image_input).convert("RGB")

            elif isinstance(image_input, Image.Image):
                logger.info("Using PIL image input.")
                image = image_input.convert("RGB")

            else:
                raise TypeError("image_input must be a file path or PIL.Image")

            image = self.test_transforms(image)
            image = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)

            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

            class_probs = {
                self.class_map[i]: float(probs[i].item())
                for i in range(len(self.class_map))
            }

            sorted_probs = dict(
                sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            )

            logger.info("ResNet prediction completed successfully.")

            return sorted_probs

        except UnidentifiedImageError:
            logger.error("Invalid image file provided to ResNet predictor.")
            raise ValueError("Invalid image file provided")

        except Exception as e:
            logger.exception("ResNet prediction failed.")
            raise RuntimeError(f"ResNet prediction failed: {str(e)}")


# ---------------- FUSION MODEL ----------------
class FusionClassifier(nn.Module):
    def __init__(self, num_classes, convnext_model_name="facebook/convnext-small-224"):
        super().__init__()

        logger.info("Initializing Fusion model architecture...")

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

        logger.info("Loading ConvNeXt backbone...")

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
            nn.Dropout(p=0.4),
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        logger.info("Fusion architecture initialized successfully.")

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

        return self.fusion_head(fused)


class FusionCarDamagePredictor:
    def __init__(self, checkpoint_path, class_map, convnext_model_name="facebook/convnext-small-224"):
        logger.info("Initializing Fusion predictor...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map

        logger.info(f"Using device for Fusion: {self.device}")

        self.eff_normalize = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        logger.info("Loading ConvNeXt image processor...")
        self.convnext_processor = ConvNextImageProcessor.from_pretrained(
            convnext_model_name
        )

        try:
            self.model = FusionClassifier(
                num_classes=len(class_map),
                convnext_model_name=convnext_model_name
            )

            logger.info(f"Loading Fusion checkpoint from: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            first_tensor = next(iter(state_dict.values()))

            if first_tensor.dtype == torch.float16:
                logger.info("FP16 checkpoint detected. Converting model to half precision.")
                self.model = self.model.half()

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            logger.info("Fusion model loaded successfully.")

        except Exception as e:
            logger.exception("Failed to load Fusion model.")
            raise RuntimeError(f"Failed to load Fusion model: {str(e)}")

    def predict(self, image_input):
        logger.info("Starting Fusion prediction...")

        try:
            if isinstance(image_input, str):
                logger.info(f"Loading image from file path: {image_input}")
                image = Image.open(image_input).convert("RGB")

            elif isinstance(image_input, Image.Image):
                logger.info("Using PIL image input.")
                image = image_input.convert("RGB")

            else:
                raise TypeError("image_input must be a file path or PIL.Image")

            pixel_eff = self.eff_normalize(image)
            pixel_eff = pixel_eff.unsqueeze(0).to(self.device)

            inputs_cnx = self.convnext_processor(
                images=image,
                return_tensors="pt"
            )

            pixel_cnx = inputs_cnx["pixel_values"].to(self.device)

            if next(self.model.parameters()).dtype == torch.float16:
                logger.info("Converting input tensors to FP16.")
                pixel_eff = pixel_eff.half()
                pixel_cnx = pixel_cnx.half()

            with torch.no_grad():
                logits = self.model(pixel_eff, pixel_cnx)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]

            class_probs = {
                self.class_map[i]: float(probs[i].item())
                for i in range(len(self.class_map))
            }

            sorted_probs = dict(
                sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            )

            logger.info("Fusion prediction completed successfully.")

            return sorted_probs

        except UnidentifiedImageError:
            logger.error("Invalid image file provided to Fusion predictor.")
            raise ValueError("Invalid image file provided")

        except Exception as e:
            logger.exception("Fusion prediction failed.")
            raise RuntimeError(f"Fusion prediction failed: {str(e)}")