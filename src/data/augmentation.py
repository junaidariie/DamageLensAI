import logging
from torchvision import transforms

from src.config import RESNET_IMAGE_SIZE, FUSION_IMAGE_SIZE

logger = logging.getLogger(__name__)


def get_resnet_train_transforms():
    logger.info("Creating ResNet training transforms...")

    return transforms.Compose([
        transforms.Resize((RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_resnet_val_transforms():
    logger.info("Creating ResNet validation transforms...")

    return transforms.Compose([
        transforms.Resize((RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_fusion_train_transforms():
    logger.info("Creating Fusion training transforms...")

    return transforms.Compose([
        transforms.Resize((FUSION_IMAGE_SIZE, FUSION_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_fusion_val_transforms():
    logger.info("Creating Fusion validation transforms...")

    return transforms.Compose([
        transforms.Resize((FUSION_IMAGE_SIZE, FUSION_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    resnet_train = get_resnet_train_transforms()
    resnet_val = get_resnet_val_transforms()
    fusion_train = get_fusion_train_transforms()
    fusion_val = get_fusion_val_transforms()

    print("\nTransforms created successfully:")
    print("ResNet Train:", resnet_train)
    print("ResNet Val:", resnet_val)
    print("Fusion Train:", fusion_train)
    print("Fusion Val:", fusion_val)