import logging
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from transformers import ConvNextImageProcessor
import onnxruntime as ort

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================================
# 1. PYTORCH RESNET MODEL & PREDICTOR (RETAINED EXCLUSIVELY FOR GRAD-CAM)
# =====================================================================
class Car_Classifier_Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        logger.info("Initializing ResNet18 PyTorch architecture for Grad-CAM...")

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


class ResnetCarDamagePredictor:
    """PyTorch Predictor wrapper used strictly by Grad-CAM for backpropagation."""
    def __init__(self, checkpoint_path, class_map):
        logger.info("Initializing PyTorch ResNet predictor for Grad-CAM...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map

        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        try:
            self.model = Car_Classifier_Resnet(num_classes=len(class_map))
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            logger.info("PyTorch ResNet model loaded successfully for Grad-CAM.")

        except Exception as e:
            logger.exception("Failed to load PyTorch ResNet model.")
            raise RuntimeError(f"Failed to load ResNet PyTorch model: {str(e)}")


# =====================================================================
# 2. ONNX RESNET PREDICTOR (FAST INFERENCE)
# =====================================================================
class ResnetONNXPredictor:
    def __init__(self, session: ort.InferenceSession, class_map: dict):
        logger.info("Initializing ResNet ONNX Predictor...")
        self.session = session
        self.class_map = class_map
        self.input_name = self.session.get_inputs()[0].name

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_input):
        logger.info("Starting ResNet ONNX prediction...")
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise TypeError("image_input must be a file path or PIL.Image")

            # Preprocess to [1, 3, 224, 224] NumPy array
            input_tensor = self.transforms(image).unsqueeze(0).numpy()

            # Execute ONNX forward pass
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0][0]

            # Softmax calculation
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            class_probs = {
                self.class_map[i]: float(probs[i])
                for i in range(len(self.class_map))
            }

            sorted_probs = dict(
                sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            )

            return sorted_probs

        except UnidentifiedImageError:
            logger.error("Invalid image file provided to ResNet ONNX predictor.")
            raise ValueError("Invalid image file provided")
        except Exception as e:
            logger.exception("ResNet ONNX prediction failed.")
            raise RuntimeError(f"ResNet ONNX prediction failed: {str(e)}")


# =====================================================================
# 3. ONNX FUSION PREDICTOR (EFFICIENTNET + CONVNEXT)
# =====================================================================
class FusionONNXPredictor:
    def __init__(self, session: ort.InferenceSession, class_map: dict, convnext_model_name="facebook/convnext-small-224"):
        logger.info("Initializing Fusion ONNX Predictor...")
        self.session = session
        self.class_map = class_map

        # Dynamically retrieve ONNX input names (EffNet & ConvNeXt)
        self.input_eff_name = self.session.get_inputs()[0].name
        self.input_cnx_name = self.session.get_inputs()[1].name

        self.eff_normalize = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.convnext_processor = ConvNextImageProcessor.from_pretrained(convnext_model_name)

    def predict(self, image_input):
        logger.info("Starting Fusion ONNX prediction...")
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise TypeError("image_input must be a file path or PIL.Image")

            # EfficientNet preprocessing -> [1, 3, 260, 260]
            pixel_eff_np = self.eff_normalize(image).unsqueeze(0).numpy()

            # ConvNeXt preprocessing -> [1, 3, 224, 224]
            inputs_cnx = self.convnext_processor(images=image, return_tensors="pt")
            pixel_cnx_np = inputs_cnx["pixel_values"].numpy()

            # Execute dual-input ONNX forward pass
            onnx_inputs = {
                self.input_eff_name: pixel_eff_np,
                self.input_cnx_name: pixel_cnx_np
            }

            outputs = self.session.run(None, onnx_inputs)
            logits = outputs[0][0]

            # Softmax calculation
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            class_probs = {
                self.class_map[i]: float(probs[i])
                for i in range(len(self.class_map))
            }

            sorted_probs = dict(
                sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            )

            return sorted_probs

        except UnidentifiedImageError:
            logger.error("Invalid image file provided to Fusion ONNX predictor.")
            raise ValueError("Invalid image file provided")
        except Exception as e:
            logger.exception("Fusion ONNX prediction failed.")
            raise RuntimeError(f"Fusion ONNX prediction failed: {str(e)}")