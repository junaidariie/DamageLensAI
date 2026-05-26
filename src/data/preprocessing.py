import logging
from collections import Counter
from sklearn.model_selection import train_test_split

from src.config import VALIDATION_SPLIT, RANDOM_SEED
from src.data.ingestion import collect_image_paths

logger = logging.getLogger(__name__)


def split_dataset(samples):
    logger.info("Starting dataset preprocessing...")

    if not samples:
        raise ValueError("Empty dataset provided.")

    image_paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    logger.info(f"Total samples before split: {len(samples)}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=VALIDATION_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    train_data = list(zip(train_paths, train_labels))
    val_data = list(zip(val_paths, val_labels))

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    logger.info(f"Train distribution: {Counter(train_labels)}")
    logger.info(f"Validation distribution: {Counter(val_labels)}")

    return train_data, val_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    samples = collect_image_paths()

    train_data, val_data = split_dataset(samples)

    print("\nTrain sample preview:")
    for sample in train_data[:5]:
        print(sample)

    print("\nValidation sample preview:")
    for sample in val_data[:5]:
        print(sample)