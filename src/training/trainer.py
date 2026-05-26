import logging
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc

        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1

            logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = val_acc
            self.counter = 0


def train_single_input_model(
    model,
    train_loader,
    eval_loader,
    optimizer,
    criterion,
    device,
    epochs,
    checkpoint_model_name,
    patience=7
):
    logger.info("Starting single-input training...")

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    early_stopping = EarlyStopping(patience=patience)

    best_acc = 0.0
    all_preds = []
    all_labels = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} Training"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()

        val_running_loss = 0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(
                eval_loader,
                desc=f"Epoch {epoch+1} Validation"
            ):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)

                loss = criterion(logits, labels)

                val_running_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(eval_loader)
        val_acc = 100 * val_correct / val_total

        logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% || "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc

            checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_model_name}.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc
                },
                checkpoint_path
            )

            logger.info(f"Best checkpoint saved at: {checkpoint_path}")

        early_stopping(val_acc)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    return all_preds, all_labels


def train_dual_input_model(
    model,
    train_loader,
    eval_loader,
    optimizer,
    criterion,
    device,
    epochs,
    checkpoint_model_name,
    patience=7
):
    logger.info("Starting dual-input training...")

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    early_stopping = EarlyStopping(patience=patience)

    best_acc = 0.0
    all_preds = []
    all_labels = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} Training"
        ):
            images_eff = batch["pixel_values_eff"].to(device)
            images_cnx = batch["pixel_values_cnx"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images_eff, images_cnx)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()

        val_running_loss = 0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                eval_loader,
                desc=f"Epoch {epoch+1} Validation"
            ):
                images_eff = batch["pixel_values_eff"].to(device)
                images_cnx = batch["pixel_values_cnx"].to(device)
                labels = batch["labels"].to(device)

                logits = model(images_eff, images_cnx)

                loss = criterion(logits, labels)

                val_running_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(eval_loader)
        val_acc = 100 * val_correct / val_total

        logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% || "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc

            checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_model_name}.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc
                },
                checkpoint_path
            )

            logger.info(f"Best checkpoint saved at: {checkpoint_path}")

        early_stopping(val_acc)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    return all_preds, all_labels


if __name__ == "__main__":
    print("Trainer utilities ready.")