import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .prediction_helper import (
    ResnetCarDamagePredictor,
    FusionCarDamagePredictor,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "resnet": {
        "repo_id": "junaid17/car-damage-classifier",
        "filename": "car-damage-classifier.pt",
    },
    "fusion": {
        "repo_id": "junaid17/best_fusion_model_fp16",
        "filename": "best_fusion_model_fp16.pt",
    },
    "yolo": {
        "repo_id": "junaid17/Yolo_Model",
        "filename": "damage_detector.pt",
    },
}


def get_checkpoint_path(model_key: str) -> Path:
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model key: {model_key}")

    config = MODEL_CONFIG[model_key]

    try:
        logger.info(f"Fetching {model_key} model from Hugging Face Hub...")
        logger.info(f"Repo: {config['repo_id']}")
        logger.info(f"File: {config['filename']}")

        local_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
        )

        logger.info(f"{model_key} model available at: {local_path}")

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
    logger.info("Starting model initialization...")

    try:
        resnet_path = get_checkpoint_path("resnet")
        fusion_path = get_checkpoint_path("fusion")
        yolo_path = get_checkpoint_path("yolo")

        logger.info("Initializing ResNet predictor...")
        resnet_predictor = ResnetCarDamagePredictor(
            checkpoint_path=resnet_path,
            class_map=class_map
        )

        logger.info("Initializing Fusion predictor...")
        fusion_predictor = FusionCarDamagePredictor(
            checkpoint_path=fusion_path,
            class_map=class_map
        )

        logger.info("Initializing YOLO model...")
        yolo_model = YOLO(str(yolo_path))

        logger.info("All models initialized successfully.")

        return resnet_predictor, fusion_predictor, yolo_model

    except Exception as e:
        logger.exception("Model initialization failed.")
        raise RuntimeError(f"Model initialization failed: {str(e)}")