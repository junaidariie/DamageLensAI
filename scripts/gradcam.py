import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class _GradCAMHook:
    __slots__ = ("activation", "gradient", "fwd_handle", "bwd_handle")

    def __init__(self, target_layer):
        self.activation = None
        self.gradient = None
        self.fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self.bwd_handle = None

    def _fwd_hook(self, module, inp, out):
        self.activation = out
        self.bwd_handle = out.register_hook(self._bwd_hook)

    def _bwd_hook(self, grad):
        self.gradient = grad

    def remove(self):
        self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()


def _postprocess_cam(cam_tensor, original_img, output_path, alpha=0.6, beta=0.4):
    h, w = original_img.height, original_img.width

    cam_min = cam_tensor.min()
    cam_max = cam_tensor.max()
    if cam_max > cam_min:
        cam_tensor = (cam_tensor - cam_min) / (cam_max - cam_min)
    else:
        cam_tensor = torch.zeros_like(cam_tensor)

    cam_np = cam_tensor.numpy()
    cam_np = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

    cam_np = np.uint8(255 * cam_np)
    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, alpha, heatmap, beta, 0)
    cv2.imwrite(output_path, overlay)


def get_resnet_gradcam(image_path, predictor, output_path):
    """Generates Grad-CAM visual heatmap using the PyTorch ResNet classifier."""
    logger.info("Starting ResNet Grad-CAM generation...")

    model = predictor.model
    model.eval()

    target_layer = model.model.layer4[-1]
    hook = _GradCAMHook(target_layer)

    try:
        original_img = Image.open(image_path).convert("RGB")
        input_tensor = predictor.test_transforms(original_img).unsqueeze(0).to(predictor.device)

        output = model(input_tensor)
        score, pred_class_idx = output[0].max(dim=0)

        logger.info(f"Predicted class index for Grad-CAM: {pred_class_idx.item()}")
        score.backward()

        if hook.activation is None or hook.gradient is None:
            raise RuntimeError("Failed to capture activations or gradients.")

        acts = hook.activation[0].detach().cpu().float()
        grads = hook.gradient[0].detach().cpu().float()

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)

        _postprocess_cam(cam, original_img, output_path)

        logger.info(f"ResNet Grad-CAM heatmap saved to: {output_path}")
        return True

    except Exception as e:
        logger.exception("ResNet Grad-CAM generation failed.")
        raise RuntimeError(f"ResNet Grad-CAM failed: {e}") from e

    finally:
        hook.remove()


def get_fusion_gradcam(image_path, predictor, output_path):
    """Fallback handler mapping to ResNet Grad-CAM (since single Grad-CAM architecture is used)."""
    logger.info("Fusion Grad-CAM requested. Directing to ResNet Grad-CAM engine...")
    return get_resnet_gradcam(image_path, predictor, output_path)