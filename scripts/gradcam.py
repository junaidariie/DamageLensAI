import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lightweight hook manager — CPU-only, no logging, direct capture
# ------------------------------------------------------------------
class _GradCAMHook:
    __slots__ = ("activation", "gradient", "fwd_handle", "bwd_handle")

    def __init__(self, target_layer):
        self.activation = None
        self.gradient = None
        self.fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self.bwd_handle = None

    def _fwd_hook(self, module, inp, out):
        self.activation = out
        # Tensor-level hook is lighter than full backward hook or retain_grad()
        self.bwd_handle = out.register_hook(self._bwd_hook)

    def _bwd_hook(self, grad):
        self.gradient = grad

    def remove(self):
        self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()


def _postprocess_cam(cam_tensor, original_img, output_path, alpha=0.5, beta=0.6):
    """
    CPU post-processing shared by both ResNet and Fusion.
    cam_tensor: 2D torch tensor [H, W] on CPU, already ReLU'd
    """
    h, w = original_img.height, original_img.width

    # Normalize on CPU (vectorized)
    cam_min = cam_tensor.min()
    cam_max = cam_tensor.max()
    if cam_max > cam_min:
        cam_tensor = (cam_tensor - cam_min) / (cam_max - cam_min)
    else:
        cam_tensor = torch.zeros_like(cam_tensor)

    # Convert to numpy once, then resize with OpenCV (very fast on CPU)
    cam_np = cam_tensor.numpy()
    cam_np = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

    cam_np = np.uint8(255 * cam_np)
    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, alpha, heatmap, beta, 0)
    cv2.imwrite(output_path, overlay)


# ------------------------------------------------------------------
# Optimized ResNet Grad-CAM (CPU)
# ------------------------------------------------------------------
def get_resnet_gradcam(image_path, predictor, output_path):
    logger.info("Starting ResNet Grad-CAM generation...")

    model = predictor.model
    model.eval()

    target_layer = model.model.layer4[-1]
    hook = _GradCAMHook(target_layer)

    try:
        original_img = Image.open(image_path).convert("RGB")
        input_tensor = predictor.test_transforms(original_img).unsqueeze(0)

        output = model(input_tensor)
        score, pred_class_idx = output[0].max(dim=0)
        pred_class_idx = pred_class_idx.item()

        logger.info(f"Predicted class index: {pred_class_idx}")
        score.backward()

        if hook.activation is None or hook.gradient is None:
            raise RuntimeError("Failed to capture activations or gradients.")

        # ----- Vectorized Grad-CAM on CPU -----
        acts = hook.activation[0].detach().float()     # [C, H, W]
        grads = hook.gradient[0].detach().float()      # [C, H, W]

        weights = grads.mean(dim=(1, 2), keepdim=True) # [C, 1, 1]
        cam = (weights * acts).sum(dim=0)              # [H, W]
        cam = F.relu(cam)

        _postprocess_cam(cam, original_img, output_path, alpha=0.6, beta=0.4)

        logger.info(f"ResNet Grad-CAM saved to: {output_path}")
        return True

    except Exception as e:
        logger.exception("ResNet Grad-CAM generation failed.")
        raise RuntimeError(f"ResNet Grad-CAM failed: {e}") from e

    finally:
        hook.remove()


# ------------------------------------------------------------------
# Optimized Fusion Grad-CAM (EfficientNet + ConvNeXt) (CPU)
# ------------------------------------------------------------------
def get_fusion_gradcam(image_path, predictor, output_path):
    logger.info("Starting Fusion Grad-CAM generation...")

    model = predictor.model
    model.eval()

    # FIX: PyTorch CPU does not support FP16 convolutions well.
    # If the model is HalfTensor, cast it to FP32 for this pass.
    is_half = next(model.parameters()).dtype == torch.float16
    if is_half:
        logger.info("FP16 model detected on CPU. Converting to FP32 for compatibility.")
        model = model.float()

    target_layer = model.eff_features[-1]
    hook = _GradCAMHook(target_layer)

    try:
        original_img = Image.open(image_path).convert("RGB")

        # CPU-only preprocessing (FloatTensor, no .to(device), no .half())
        pixel_eff = predictor.eff_normalize(original_img).unsqueeze(0)
        pixel_cnx = predictor.convnext_processor(
            images=original_img, return_tensors="pt"
        )["pixel_values"]

        output = model(pixel_eff, pixel_cnx)
        score, pred_class_idx = output[0].max(dim=0)
        pred_class_idx = pred_class_idx.item()

        logger.info(f"Predicted class index: {pred_class_idx}")
        score.backward()

        if hook.activation is None or hook.gradient is None:
            raise RuntimeError("Failed to capture activations or gradients.")

        # ----- Vectorized Grad-CAM on CPU -----
        acts = hook.activation[0].detach().float()     # [C, H, W]
        grads = hook.gradient[0].detach().float()        # [C, H, W]

        weights = grads.mean(dim=(1, 2), keepdim=True) # [C, 1, 1]
        cam = (weights * acts).sum(dim=0)              # [H, W]
        cam = F.relu(cam)

        _postprocess_cam(cam, original_img, output_path, alpha=0.5, beta=0.6)

        logger.info(f"Fusion Grad-CAM saved to: {output_path}")
        return True

    except Exception as e:
        logger.exception("Fusion Grad-CAM generation failed.")
        raise RuntimeError(f"Fusion Grad-CAM failed: {e}") from e

    finally:
        hook.remove()