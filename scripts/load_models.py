import logging
from pathlib import Path
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .prediction_helper import ResnetCarDamagePredictor

logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "resnet_onnx": {
        "repo_id": "junaid17/car-damage-classifier",
        "filename": "car-damage-classifier.onnx",
    },
    "fusion_onnx": {
        "repo_id": "junaid17/best_fusion_model_fp16",
        "filename": "fusion_model.onnx",
    },
    "yolo_onnx": {
        "repo_id": "junaid17/Yolo_Model",
        "filename": "damage_detector.onnx",
    },
    "resnet_pt": {
        "repo_id": "junaid17/car-damage-classifier",
        "filename": "car-damage-classifier.pt",
    },
}


def get_checkpoint_path(model_key: str) -> Path:
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model key: {model_key}")

    config = MODEL_CONFIG[model_key]

    try:
        logger.info(f"Fetching {model_key} model from Hugging Face Hub...")
        logger.info(f"Repo: {config['repo_id']} | File: {config['filename']}")

        local_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
        )

        logger.info(f"{model_key} model downloaded to: {local_path}")
        return Path(local_path)

    except Exception as e:
        logger.exception(f"Failed to fetch {model_key} model.")
        raise RuntimeError(f"Failed to load {model_key} checkpoint: {str(e)}")


class ModelLoader:
    def __init__(self):
        logger.info("Initializing ModelLoader...")

    def get_model_path(self, model_key: str) -> Path:
        return get_checkpoint_path(model_key)


def initialize_models(class_map):
    logger.info("Starting model initialization pipeline...")

    try:
        # 1. Download/Fetch all 4 model file paths
        resnet_onnx_path = get_checkpoint_path("resnet_onnx")
        fusion_onnx_path = get_checkpoint_path("fusion_onnx")
        yolo_onnx_path   = get_checkpoint_path("yolo_onnx")
        resnet_pt_path   = get_checkpoint_path("resnet_pt")

        # Define ONNX Execution Providers (GPU if available, fallback to CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # 2. Initialize ONNX Runtime Sessions for Classifier Models
        logger.info("Initializing ResNet ONNX Session...")
        resnet_onnx_session = ort.InferenceSession(str(resnet_onnx_path), providers=providers)

        logger.info("Initializing Fusion ONNX Session...")
        fusion_onnx_session = ort.InferenceSession(str(fusion_onnx_path), providers=providers)

        # 3. Initialize YOLO ONNX Model via Ultralytics (enables ONNX runtime with .predict API)
        logger.info("Initializing YOLO ONNX Model via Ultralytics...")
        yolo_onnx_model = YOLO(str(yolo_onnx_path), task="detect")

        # 4. Initialize PyTorch ResNet Predictor (specifically reserved for Grad-CAM)
        logger.info("Initializing PyTorch ResNet model for Grad-CAM...")
        resnet_gradcam_predictor = ResnetCarDamagePredictor(
            checkpoint_path=resnet_pt_path,
            class_map=class_map
        )

        logger.info("All 4 models (3 ONNX + 1 PyTorch Grad-CAM) initialized successfully.")

        return {
            "resnet_onnx": resnet_onnx_session,
            "fusion_onnx": fusion_onnx_session,
            "yolo_onnx": yolo_onnx_model,
            "resnet_pt": resnet_gradcam_predictor,
        }

    except Exception as e:
        logger.exception("Model initialization failed.")
        raise RuntimeError(f"Model initialization failed: {str(e)}")