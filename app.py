import os
import uuid
import shutil
import logging
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from scripts.gradcam import get_resnet_gradcam
from scripts.yolo_predict import get_yolo_damage_boxes
from scripts.load_models import initialize_models
from scripts.prediction_helper import ResnetONNXPredictor, FusionONNXPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

CLASS_MAP = {
    0: "Front Breakage",
    1: "Front Crushed",
    2: "Front Normal",
    3: "Rear Breakage",
    4: "Rear Crushed",
    5: "Rear Normal"
}

# Unified dictionary to hold all loaded models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models at startup...")
    try:
        models = initialize_models(CLASS_MAP)
        ml_models["resnet_onnx"] = ResnetONNXPredictor(models["resnet_onnx"], CLASS_MAP)
        ml_models["fusion_onnx"] = FusionONNXPredictor(models["fusion_onnx"], CLASS_MAP)
        ml_models["resnet_pt"] = models["resnet_pt"]
        ml_models["yolo_onnx"] = models["yolo_onnx"]
        
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.exception("Model loading failed.")
        raise RuntimeError(str(e))
    
    yield
    ml_models.clear()
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

def validate_image(upload_file: UploadFile):
    if not upload_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

def save_upload(upload_file: UploadFile):
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_input.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return unique_id, filename, file_path

# Endpoints
@app.get("/")
def api_status():
    return {"status": "API is running"}

@app.post("/predict/resnet")
async def resnet_prediction(file: UploadFile = File(...)):
    validate_image(file)
    try:
        pil_image = Image.open(file.file).convert("RGB")
        result = ml_models["resnet_onnx"].predict(pil_image)
        return {"status": "success", "prediction": result}
    except Exception as e:
        logger.exception("ResNet ONNX prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/fusion")
async def fusion_prediction(file: UploadFile = File(...)):
    validate_image(file)
    try:
        pil_image = Image.open(file.file).convert("RGB")
        result = ml_models["fusion_onnx"].predict(pil_image)
        return {"status": "success", "prediction": result}
    except Exception as e:
        logger.exception("Fusion ONNX prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/yolo")
async def yolo_detection(file: UploadFile = File(...)):
    validate_image(file)
    try:
        unique_id, input_filename, input_path = save_upload(file)
        output_name = f"{unique_id}_yolo.jpg"
        output_path = os.path.join(RESULT_DIR, output_name)

        result = get_yolo_damage_boxes(input_path, ml_models["yolo_onnx"], output_path)

        return {
            "status": "success",
            "original_image": f"/static/uploads/{input_filename}",
            "yolo_image": f"/static/results/{output_name}",
            "detections": result["detections"],
            "total_detections": result["total_detections"],
            "message": result["message"]
        }
    except Exception as e:
        logger.exception("YOLO ONNX detection failed.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/gradcam")
async def gradcam_generation(file: UploadFile = File(...)):
    validate_image(file)
    try:
        unique_id, input_filename, input_path = save_upload(file)
        output_name = f"{unique_id}_gradcam.jpg"
        output_path = os.path.join(RESULT_DIR, output_name)

        get_resnet_gradcam(input_path, ml_models["resnet_pt"], output_path)

        return {
            "status": "success",
            "original_image": f"/static/uploads/{input_filename}",
            "gradcam_image": f"/static/results/{output_name}"
        }
    except Exception as e:
        logger.exception("Grad-CAM generation failed.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/comprehensive")
async def comprehensive_prediction(
    file: UploadFile = File(...),
    mode: str = Query("fusion", description="Classification model to use: 'resnet' or 'fusion'")
):
    validate_image(file)
    mode = mode.lower()

    if mode not in {"resnet", "fusion"}:
        raise HTTPException(status_code=400, detail="mode must be 'resnet' or 'fusion'")

    try:
        unique_id, input_filename, input_path = save_upload(file)
        pil_image = Image.open(input_path).convert("RGB")

        # 1. Classification
        if mode == "resnet":
            classification_result = ml_models["resnet_onnx"].predict(pil_image)
        else:
            classification_result = ml_models["fusion_onnx"].predict(pil_image)

        # 2. YOLO Bounding Boxes
        yolo_output_name = f"{unique_id}_yolo.jpg"
        yolo_output_path = os.path.join(RESULT_DIR, yolo_output_name)
        yolo_result = get_yolo_damage_boxes(input_path, ml_models["yolo_onnx"], yolo_output_path)

        # 3. Grad-CAM
        gradcam_output_name = f"{unique_id}_gradcam.jpg"
        gradcam_output_path = os.path.join(RESULT_DIR, gradcam_output_name)
        get_resnet_gradcam(input_path, ml_models["resnet_pt"], gradcam_output_path)

        return {
            "status": "success",
            "mode": mode,
            "original_image": f"/static/uploads/{input_filename}",
            "classification": classification_result,
            "yolo": {
                "image": f"/static/results/{yolo_output_name}",
                "detections": yolo_result["detections"],
                "total_detections": yolo_result["total_detections"]
            },
            "gradcam": f"/static/results/{gradcam_output_name}"
        }

    except Exception as e:
        logger.exception("Comprehensive prediction pipeline failed.")
        raise HTTPException(status_code=500, detail=str(e))