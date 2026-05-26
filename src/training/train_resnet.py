import logging
import torch.nn as nn
from torch.optim import AdamW

from src.config import DEVICE, EPOCHS, NUM_CLASSES
from src.models.resnet_model import CarClassifierResNet
from src.data.dataset import create_resnet_dataloaders
from src.training.trainer import train_single_input_model

logger = logging.getLogger(__name__)


def run_resnet_training():
    logger.info("Initializing ResNet training pipeline...")

    train_loader, eval_loader = create_resnet_dataloaders()

    model = CarClassifierResNet(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW([
        {
            "params": model.model.layer3.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.model.layer4.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.model.fc.parameters(),
            "lr": 1e-4
        }
    ])

    logger.info("Starting ResNet training...")

    all_preds, all_labels = train_single_input_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        epochs=EPOCHS,
        checkpoint_model_name="best_resnet_model",
        patience=7
    )

    logger.info("ResNet training completed.")

    return all_preds, all_labels


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    preds, labels = run_resnet_training()

    print("\nTraining completed successfully.")
    print("Prediction samples:", preds[:10])
    print("Label samples:", labels[:10])