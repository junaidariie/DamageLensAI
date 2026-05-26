import logging

from src.config import (
    BASE_DIR,
    CHECKPOINT_DIR,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    NUM_CLASSES
)

logger = logging.getLogger(__name__)


def test_config():
    logger.info("Testing config settings...")

    assert BASE_DIR.exists(), "BASE_DIR missing"
    assert CHECKPOINT_DIR.exists(), "CHECKPOINT_DIR missing"

    assert DEVICE in ["cpu", "cuda"], "Invalid device"
    assert BATCH_SIZE > 0, "Invalid batch size"
    assert EPOCHS > 0, "Invalid epochs"
    assert NUM_CLASSES == 6, "NUM_CLASSES mismatch"

    logger.info("Config test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_config()

    print("Config test completed successfully.")