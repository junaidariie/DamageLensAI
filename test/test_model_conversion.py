import logging
import os

from src.export.conver_model import convert_fusion_to_fp16
from src.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def test_model_conversion():
    logger.info("Testing fusion FP16 conversion...")

    input_checkpoint = CHECKPOINT_DIR / "best_fusion_model.pt"

    assert input_checkpoint.exists(), \
        f"Missing checkpoint: {input_checkpoint}"

    output_path = convert_fusion_to_fp16()

    assert output_path.exists(), \
        "FP16 model was not created"

    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    assert size_mb > 0, \
        "Generated FP16 model is empty"

    logger.info("Model conversion test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_model_conversion()

    print("Model conversion test completed successfully.")