import logging

from src.data.ingestion import collect_image_paths
from src.data.preprocessing import split_dataset

logger = logging.getLogger(__name__)


def test_preprocessing():
    logger.info("Testing preprocessing...")

    samples = collect_image_paths()

    train_data, val_data = split_dataset(samples)

    assert len(train_data) > 0, "Train split is empty"
    assert len(val_data) > 0, "Validation split is empty"

    train_paths = set(x[0] for x in train_data)
    val_paths = set(x[0] for x in val_data)

    overlap = train_paths.intersection(val_paths)

    assert len(overlap) == 0, "Train and validation overlap found"

    logger.info("Preprocessing test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_preprocessing()

    print("Preprocessing test completed successfully.")