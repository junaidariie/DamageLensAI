import logging
import torch

from src.models.fusion_model import FusionClassifier
from src.config import NUM_CLASSES

logger = logging.getLogger(__name__)


def test_fusion_model():
    logger.info("Testing Fusion model architecture...")

    model = FusionClassifier(
        num_classes=NUM_CLASSES
    )

    model.eval()

    eff_dummy = torch.randn(2, 3, 260, 260)
    cnx_dummy = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(
            eff_dummy,
            cnx_dummy
        )

    assert output.shape == (2, NUM_CLASSES), \
        f"Unexpected output shape: {output.shape}"

    logger.info("Fusion model test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_fusion_model()

    print("Fusion model test completed successfully.")