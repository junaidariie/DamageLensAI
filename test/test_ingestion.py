import logging
import os

from src.data.ingestion import collect_image_paths

logger = logging.getLogger(__name__)


def test_ingestion():
    logger.info("Testing ingestion...")

    samples = collect_image_paths()

    assert len(samples) > 0, "No samples found"

    image_path, label = samples[0]

    assert os.path.exists(image_path), "Image path missing"
    assert isinstance(label, int), "Label invalid"

    logger.info("Ingestion test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_ingestion()

    print("Ingestion test completed successfully.")