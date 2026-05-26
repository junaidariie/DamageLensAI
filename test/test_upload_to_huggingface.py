import logging
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.export.upload_to_huggingface import MODELS

logger = logging.getLogger(__name__)


def test_huggingface_upload_setup():
    logger.info("Testing Hugging Face upload setup...")

    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    assert hf_token is not None, \
        "HF_TOKEN missing in .env"

    assert hf_token.startswith("hf_"), \
        "Invalid Hugging Face token format"

    api = HfApi(token=hf_token)

    assert api is not None, \
        "Failed to initialize Hugging Face API"

    for repo_name, model_info in MODELS.items():
        file_path = model_info["path"]
        filename = model_info["filename"]

        assert file_path.exists(), \
            f"Missing model file: {file_path}"

        assert filename.endswith(".pt"), \
            f"Invalid model filename: {filename}"

        assert repo_name.startswith("new-"), \
            f"Repo naming invalid: {repo_name}"

    logger.info("Hugging Face upload setup test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_huggingface_upload_setup()

    print("Hugging Face upload test completed successfully.")