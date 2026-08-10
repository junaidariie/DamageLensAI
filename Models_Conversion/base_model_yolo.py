import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


def load_yolo_model_from_hf(
    repo_id: str = "junaid17/Yolo_Model",
    filename: str = "damage_detector.pt",
    hf_token: str = None,
) -> YOLO:
    """
    Downloads and loads a trained Ultralytics YOLO model from Hugging Face Hub.
    """
    print(f"Downloading YOLO checkpoint '{filename}' from Hugging Face repo '{repo_id}'...")
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=hf_token
    )
    model = YOLO(checkpoint_path)
    print("✅ YOLO model loaded successfully.")
    return model