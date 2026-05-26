import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ConvNextImageProcessor

from src.config import (
    BATCH_SIZE,
    NUM_WORKERS
)

from src.data.ingestion import collect_image_paths
from src.data.preprocessing import split_dataset
from src.data.augmentation import (
    get_resnet_train_transforms,
    get_resnet_val_transforms,
    get_fusion_train_transforms,
    get_fusion_val_transforms
)

logger = logging.getLogger(__name__)


class ResNetDataset(Dataset):
    def __init__(self, samples, transforms=None):
        self.samples = samples
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, label


class FusionDataset(Dataset):
    def __init__(
        self,
        samples,
        transforms=None,
        convnext_model_name="facebook/convnext-small-224"
    ):
        self.samples = samples
        self.transforms = transforms

        logger.info("Loading ConvNeXt processor...")

        self.processor = ConvNextImageProcessor.from_pretrained(
            convnext_model_name
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            eff_tensor = self.transforms(image)
        else:
            raise ValueError("Fusion transforms are required.")

        convnext_inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        convnext_tensor = convnext_inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values_eff": eff_tensor,
            "pixel_values_cnx": convnext_tensor,
            "labels": label
        }


def create_resnet_dataloaders():
    logger.info("Creating ResNet dataloaders...")

    samples = collect_image_paths()
    train_data, val_data = split_dataset(samples)

    train_dataset = ResNetDataset(
        train_data,
        transforms=get_resnet_train_transforms()
    )

    val_dataset = ResNetDataset(
        val_data,
        transforms=get_resnet_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    logger.info("ResNet dataloaders created successfully.")

    return train_loader, val_loader


def create_fusion_dataloaders():
    logger.info("Creating Fusion dataloaders...")

    samples = collect_image_paths()
    train_data, val_data = split_dataset(samples)

    train_dataset = FusionDataset(
        train_data,
        transforms=get_fusion_train_transforms()
    )

    val_dataset = FusionDataset(
        val_data,
        transforms=get_fusion_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    logger.info("Fusion dataloaders created successfully.")

    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("\nTesting ResNet dataloaders...\n")

    train_loader, val_loader = create_resnet_dataloaders()

    images, labels = next(iter(train_loader))

    print("ResNet batch shape:", images.shape)
    print("ResNet labels shape:", labels.shape)

    print("\nTesting Fusion dataloaders...\n")

    train_loader, val_loader = create_fusion_dataloaders()

    batch = next(iter(train_loader))

    print(
        "Fusion EfficientNet batch shape:",
        batch["pixel_values_eff"].shape
    )

    print(
        "Fusion ConvNeXt batch shape:",
        batch["pixel_values_cnx"].shape
    )

    print(
        "Fusion labels shape:",
        batch["labels"].shape
    )