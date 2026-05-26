import logging
import torch.nn as nn
from torch.optim import AdamW

from src.config import DEVICE, EPOCHS, NUM_CLASSES
from src.models.fusion_model import FusionClassifier
from src.data.dataset import create_fusion_dataloaders
from src.training.trainer import train_dual_input_model

logger = logging.getLogger(__name__)


def run_fusion_training():
    logger.info("Initializing Fusion training pipeline...")

    train_loader, eval_loader = create_fusion_dataloaders()

    model = FusionClassifier(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = AdamW([
        # EfficientNet unfrozen blocks
        {
            "params": model.eff_features[5].parameters(),
            "lr": 1e-5
        },
        {
            "params": model.eff_features[6].parameters(),
            "lr": 3e-5
        },
        {
            "params": model.eff_features[7].parameters(),
            "lr": 3e-5
        },

        # ConvNeXt unfrozen blocks
        {
            "params": model.cnx_backbone.encoder.stages[2].parameters(),
            "lr": 3e-5
        },
        {
            "params": model.cnx_backbone.encoder.stages[3].parameters(),
            "lr": 3e-5
        },
        {
            "params": model.cnx_backbone.layernorm.parameters(),
            "lr": 3e-5
        },

        # Fusion head
        {
            "params": model.fusion_head.parameters(),
            "lr": 1e-4
        }

    ], weight_decay=1e-4)

    logger.info("Starting Fusion training...")

    all_preds, all_labels = train_dual_input_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epochs=EPOCHS,
        checkpoint_model_name="best_fusion_model",
        patience=7
    )

    logger.info("Fusion training completed.")

    return all_preds, all_labels


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    preds, labels = run_fusion_training()

    print("\nFusion training completed successfully.")
    print("Prediction samples:", preds[:10])
    print("Label samples:", labels[:10])