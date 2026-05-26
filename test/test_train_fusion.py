import logging

from src.training.train_fusion import run_fusion_training
from src.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def test_train_fusion():
    logger.info("Testing Fusion training pipeline...")

    checkpoint_path = CHECKPOINT_DIR / "best_fusion_model.pt"

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    preds, labels = run_fusion_training()

    assert checkpoint_path.exists(), \
        "Fusion checkpoint was not created"

    assert len(preds) > 0, \
        "No predictions returned"

    assert len(labels) > 0, \
        "No labels returned"

    logger.info("Fusion training test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_train_fusion()

    print("Fusion training test completed successfully.")