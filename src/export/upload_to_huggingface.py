import os
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.config import CHECKPOINT_DIR

load_dotenv()

logger = logging.getLogger(__name__)

HF_USERNAME = "junaid17"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found in .env file."
    )

MODELS = {
    "new-damagelens-resnet-classifier": {
        "path": CHECKPOINT_DIR / "best_resnet_model.pt",
        "filename": "new_best_resnet_model.pt"
    },

    "new-damagelens-fusion-fp16": {
        "path": CHECKPOINT_DIR / "best_fusion_model_fp16.pt",
        "filename": "new_best_fusion_model_fp16.pt"
    },

    "new-damagelens-yolo-detector": {
        "path": CHECKPOINT_DIR / "damage_detector.pt",
        "filename": "new_damage_detector.pt"
    }
}


def upload_model(api, repo_name, file_path, filename):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {file_path}"
        )

    repo_id = f"{HF_USERNAME}/{repo_name}"

    logger.info(f"Creating repo: {repo_id}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True
    )

    logger.info(f"Uploading {filename} to {repo_id}")

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model"
    )

    logger.info(f"Upload completed: {repo_id}")


def upload_all_models():
    logger.info("Starting Hugging Face model uploads...")

    api = HfApi(token=HF_TOKEN)

    for repo_name, model_info in MODELS.items():
        upload_model(
            api=api,
            repo_name=repo_name,
            file_path=model_info["path"],
            filename=model_info["filename"]
        )

    logger.info("All model uploads completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    upload_all_models()

    print("\nAll models uploaded successfully.")