import logging

from src.data.dataset import (
    create_resnet_dataloaders,
    create_fusion_dataloaders
)

logger = logging.getLogger(__name__)


def test_dataset():
    logger.info("Testing dataset loaders...")

    # ---------------- ResNet ----------------
    resnet_loader, _ = create_resnet_dataloaders()

    images, labels = next(iter(resnet_loader))

    assert images.shape[1:] == (3, 128, 128), \
        f"Unexpected ResNet image shape: {images.shape}"

    assert len(labels.shape) == 1, \
        f"Unexpected ResNet labels shape: {labels.shape}"

    logger.info("ResNet dataloader test passed.")

    # ---------------- Fusion ----------------
    fusion_loader, _ = create_fusion_dataloaders()

    batch = next(iter(fusion_loader))

    assert "pixel_values_eff" in batch, "Missing EfficientNet input"
    assert "pixel_values_cnx" in batch, "Missing ConvNeXt input"
    assert "labels" in batch, "Missing labels"

    assert batch["pixel_values_eff"].shape[1:] == (3, 260, 260), \
        f"Unexpected Fusion EfficientNet shape: {batch['pixel_values_eff'].shape}"

    assert batch["pixel_values_cnx"].shape[1:] == (3, 224, 224), \
        f"Unexpected Fusion ConvNeXt shape: {batch['pixel_values_cnx'].shape}"

    assert len(batch["labels"].shape) == 1, \
        f"Unexpected Fusion labels shape: {batch['labels'].shape}"

    logger.info("Fusion dataloader test passed.")
    logger.info("Dataset test passed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_dataset()

    print("Dataset test completed successfully.")