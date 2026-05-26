import logging
from shutil import copy2, rmtree
from ultralytics import YOLO

from src.config import BASE_DIR, CHECKPOINT_DIR, DEVICE

logger = logging.getLogger(__name__)

YOLO_DATASET_CONFIG = BASE_DIR / "data" / "yolo" / "dataset_custom.yaml"
YOLO_BASE_MODEL = CHECKPOINT_DIR / "yolo11m.pt"


def run_yolo_training():
    logger.info("Initializing YOLO training pipeline...")

    if not YOLO_DATASET_CONFIG.exists():
        raise FileNotFoundError(
            f"YOLO dataset config not found: {YOLO_DATASET_CONFIG}"
        )

    if not YOLO_BASE_MODEL.exists():
        raise FileNotFoundError(
            f"YOLO base model not found: {YOLO_BASE_MODEL}"
        )

    yolo_device = 0 if DEVICE == "cuda" else "cpu"

    checkpoint_root = CHECKPOINT_DIR.resolve()
    temp_run_name = "temp_yolo_run"

    logger.info("Loading YOLO base model...")

    model = YOLO(str(YOLO_BASE_MODEL.resolve()))

    logger.info("Starting YOLO training...")

    model.train(
        data=str(YOLO_DATASET_CONFIG.resolve()),
        imgsz=416,
        batch=4,
        epochs=1,
        device=yolo_device,
        project=str(checkpoint_root),
        name=temp_run_name,
        exist_ok=True
    )

    best_model_path = (
        checkpoint_root /
        temp_run_name /
        "weights" /
        "best.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"YOLO best model not found: {best_model_path}"
        )

    final_model_path = checkpoint_root / "damage_detector.pt"

    copy2(best_model_path, final_model_path)

    logger.info(f"Final YOLO model saved at: {final_model_path}")

    # cleanup temp training folder
    temp_run_dir = checkpoint_root / temp_run_name

    if temp_run_dir.exists():
        rmtree(temp_run_dir)
        logger.info("Temporary YOLO training artifacts deleted.")

    return final_model_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    model_path = run_yolo_training()

    print("\nYOLO training completed successfully.")
    print(f"Saved model: {model_path}")