import os
import logging
import torch

from src.config import DEVICE, NUM_CLASSES, CHECKPOINT_DIR
from src.models.fusion_model import FusionClassifier

logger = logging.getLogger(__name__)

INPUT_CHECKPOINT = CHECKPOINT_DIR / "best_fusion_model.pt"
OUTPUT_CHECKPOINT = CHECKPOINT_DIR / "best_fusion_model_fp16.pt"


def convert_fusion_to_fp16():
    logger.info("Initializing Fusion model for FP16 conversion...")

    if not INPUT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {INPUT_CHECKPOINT}"
        )

    model = FusionClassifier(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    logger.info(f"Loading checkpoint from: {INPUT_CHECKPOINT}")

    checkpoint = torch.load(
        INPUT_CHECKPOINT,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info("Model weights loaded successfully.")

    model.eval()

    logger.info("Converting model to FP16...")

    model = model.half()

    torch.save(
        model.state_dict(),
        OUTPUT_CHECKPOINT
    )

    size_mb = os.path.getsize(OUTPUT_CHECKPOINT) / (1024 * 1024)

    logger.info(f"FP16 model saved at: {OUTPUT_CHECKPOINT}")
    logger.info(f"FP16 model size: {size_mb:.2f} MB")

    return OUTPUT_CHECKPOINT


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    fp16_path = convert_fusion_to_fp16()

    print("\nFusion FP16 conversion completed successfully.")
    print(f"Saved model: {fp16_path}")