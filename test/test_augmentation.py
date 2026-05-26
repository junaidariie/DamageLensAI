import logging
from PIL import Image

from src.data.augmentation import (
    get_resnet_train_transforms,
    get_fusion_train_transforms
)

logger = logging.getLogger(__name__)


def test_augmentation():
    logger.info("Testing augmentation pipelines...")

    dummy_image = Image.new("RGB", (300, 300))

    resnet_transform = get_resnet_train_transforms()
    fusion_transform = get_fusion_train_transforms()

    resnet_tensor = resnet_transform(dummy_image)
    fusion_tensor = fusion_transform(dummy_image)

    assert resnet_tensor.shape == (3, 128, 128), \
        f"Unexpected ResNet shape: {resnet_tensor.shape}"

    assert fusion_tensor.shape == (3, 260, 260), \
        f"Unexpected Fusion shape: {fusion_tensor.shape}"

    logger.info("Augmentation test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_augmentation()

    print("Augmentation test completed successfully.")