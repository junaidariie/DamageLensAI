import logging
from pathlib import Path

from src.config import DATASET_DIR, CLASS_TO_IDX

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_paths():
    logger.info("Starting dataset ingestion...")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    samples = []

    for class_name, label in CLASS_TO_IDX.items():
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            logger.warning(f"Missing class folder: {class_dir}")
            continue

        image_count = 0

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((str(image_path), label))
                image_count += 1

        logger.info(f"{class_name}: {image_count} images found")

    if not samples:
        raise ValueError("No valid images found in dataset.")

    logger.info(f"Total images collected: {len(samples)}")

    return samples


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data = collect_image_paths()

    print(f"\nTotal samples: {len(data)}")
    print("First 5 samples:")

    for sample in data[:5]:
        print(sample)