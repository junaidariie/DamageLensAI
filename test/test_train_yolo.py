import logging

from src.training.train_yolo import run_yolo_training
from src.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def test_train_yolo():
    logger.info("Testing YOLO training pipeline...")

    checkpoint_path = CHECKPOINT_DIR / "damage_detector.pt"

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    output_path = run_yolo_training()

    assert checkpoint_path.exists(), \
        "YOLO checkpoint was not created"

    assert output_path.exists(), \
        "Returned YOLO model path invalid"

    logger.info("YOLO training test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_train_yolo()

    print("YOLO training test completed successfully.")