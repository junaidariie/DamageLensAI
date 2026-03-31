import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from transformers import DeiTForImageClassification, DeiTImageProcessor

# ================================ ResNet-18 Classifier ================================
class Car_Classifier_Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")

        # Freeze most layers
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC head
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
    def __init__(self, checkpoint_path, class_map):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map

        self.test_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        try:
            self.model = Car_Classifier_Resnet(num_classes=len(class_map))
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet model: {str(e)}")

    def resnet_predict(self, image_input):
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
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
            return dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

        except UnidentifiedImageError:
            raise ValueError("Invalid image file provided")
        except Exception as e:
            raise RuntimeError(f"ResNet prediction failed: {str(e)}")

# ================================ DeiT Classifier ================================
class DeitCarDamagePredictor:
    def __init__(self, checkpoint_path, class_map):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = class_map
        self.checkpoint_path = checkpoint_path

        self.transform = transforms.Compose([transforms.Resize((224, 224))])
        model_name = "facebook/deit-base-distilled-patch16-224"

        try:
            self.processor = DeiTImageProcessor.from_pretrained(model_name)
            self.model = DeiTForImageClassification.from_pretrained(
                model_name,
                num_labels=len(class_map),
                ignore_mismatched_sizes=True
            )
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DeiT model: {str(e)}")

    def deit_predict(self, image_input):
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise TypeError("image_input must be a file path or PIL.Image")

            image = self.transform(image)
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            class_probs = {
                self.class_map[i]: float(probs[i].item())
                for i in range(len(self.class_map))
            }
            return dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

        except UnidentifiedImageError:
            raise ValueError("Invalid image file provided")
        except Exception as e:
            raise RuntimeError(f"DeiT prediction failed: {str(e)}")

# ================================ Fusion Predictor ================================
class FusionCarDamagePredictor:
    def __init__(self, resnet_predictor, deit_predictor, resnet_weight=0.5, deit_weight=0.5):
        if resnet_weight < 0 or deit_weight < 0:
            raise ValueError("Weights must be non-negative")
        total = resnet_weight + deit_weight
        if total == 0:
            raise ValueError("At least one weight must be greater than 0")

        self.resnet_predictor = resnet_predictor
        self.deit_predictor = deit_predictor
        self.resnet_weight = resnet_weight / total
        self.deit_weight = deit_weight / total

    def fuse_predict(self, image_input):
        try:
            resnet_output = self.resnet_predictor.resnet_predict(image_input)
            deit_output = self.deit_predictor.deit_predict(image_input)

            all_classes = set(resnet_output.keys()).union(set(deit_output.keys()))
            fused_output = {}
            for cls in all_classes:
                resnet_prob = resnet_output.get(cls, 0.0)
                deit_prob = deit_output.get(cls, 0.0)
                fused_prob = self.resnet_weight * resnet_prob + self.deit_weight * deit_prob
                fused_output[cls] = float(fused_prob)

            fused_output = dict(sorted(fused_output.items(), key=lambda x: x[1], reverse=True))
            final_class = next(iter(fused_output))
            final_confidence = fused_output[final_class]

            return {
                "resnet_output": resnet_output,
                "deit_output": deit_output,
                "fused_output": fused_output,
                "final_prediction": final_class,
                "final_confidence": round(final_confidence, 4)
            }
        except Exception as e:
            raise RuntimeError(f"Fusion prediction failed: {str(e)}")