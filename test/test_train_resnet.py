import logging

from src.training.train_resnet import run_resnet_training
from src.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def test_train_resnet():
    logger.info("Testing ResNet training pipeline...")

    checkpoint_path = CHECKPOINT_DIR / "best_resnet_model.pt"

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    preds, labels = run_resnet_training()

    assert checkpoint_path.exists(), \
        "ResNet checkpoint was not created"

    assert len(preds) > 0, \
        "No predictions returned"

    assert len(labels) > 0, \
        "No labels returned"

    logger.info("ResNet training test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_train_resnet()

    print("ResNet training test completed successfully.")