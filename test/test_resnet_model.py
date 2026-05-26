import logging
import torch

from src.models.resnet_model import CarClassifierResNet
from src.config import NUM_CLASSES

logger = logging.getLogger(__name__)


def test_resnet_model():
    logger.info("Testing ResNet model architecture...")

    model = CarClassifierResNet(
        num_classes=NUM_CLASSES
    )

    model.eval()

    dummy_input = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, NUM_CLASSES), \
        f"Unexpected output shape: {output.shape}"

    logger.info("ResNet model test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_resnet_model()

    print("ResNet model test completed successfully.")