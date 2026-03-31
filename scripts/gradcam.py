import cv2
import numpy as np
from PIL import Image

def get_resnet_gradcam(image_path, predictor, output_path):
    model = predictor.model
    device = predictor.device
    model.eval()

    features, gradients = [], []

    def forward_hook(module, input, output): features.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    target_layer = model.model.layer4[-1]
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    original_img = Image.open(image_path).convert("RGB")
    input_tensor = predictor.test_transforms(original_img).unsqueeze(0).to(device)

    model.zero_grad()
    output = model(input_tensor)
    pred_class_idx = output.argmax(dim=1).item()
    
    score = output[0, pred_class_idx]
    score.backward()

    handle_fw.remove()
    handle_bw.remove()

    acts = features[0].cpu().data.numpy()[0]
    grads = gradients[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_img.width, original_img.height))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_np = np.array(original_img)
    
    # Overlay logic (OpenCV style)
    overlay = cv2.addWeighted(cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)
    return True

def get_deit_gradcam(image_path, predictor, output_path):
    model = predictor.model
    processor = predictor.processor
    device = predictor.device
    model.eval()

    features, gradients = [], []

    def forward_hook(module, input, output): features.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    target_layer = model.deit.encoder.layer[-1].layernorm_before
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    original_img = Image.open(image_path).convert("RGB")
    inputs = processor(images=original_img, return_tensors="pt").to(device)

    model.zero_grad()
    outputs = model(**inputs)
    pred_class_idx = outputs.logits.argmax(dim=1).item()
    
    score = outputs.logits[0, pred_class_idx]
    score.backward()

    handle_fw.remove()
    handle_bw.remove()

    acts = features[0].cpu().data.numpy()[0]
    grads = gradients[0].cpu().data.numpy()[0]
    cam = np.sum(grads * acts, axis=-1) 
    cam = cam[2:] # Remove CLS and Distillation tokens

    grid_size = int(np.sqrt(cam.shape[0]))
    cam = cam.reshape(grid_size, grid_size)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_img.width, original_img.height))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original_np = np.array(original_img)
    
    overlay = cv2.addWeighted(cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)
    return True