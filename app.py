import os
import uuid
import shutil
import logging
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from scripts.gradcam import get_resnet_gradcam, get_fusion_gradcam
from scripts.yolo_predict import get_yolo_damage_boxes
from scripts.load_models import initialize_models

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------- ENV ----------------
load_dotenv()

# ---------------- DIRECTORIES ----------------
UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- GLOBAL MODELS ----------------
resnet_predictor = None
fusion_predictor = None
yolo_model = None

CLASS_MAP = {
    0: "Front Breakage",
    1: "Front Crushed",
    2: "Front Normal",
    3: "Rear Breakage",
    4: "Rear Crushed",
    5: "Rear Normal"
}

# ---------------- FASTAPI STARTUP ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global resnet_predictor, fusion_predictor, yolo_model

    logger.info("Loading models at startup...")

    try:
        resnet_predictor, fusion_predictor, yolo_model = initialize_models(CLASS_MAP)
        logger.info("All models loaded successfully.")

    except Exception as e:
        logger.exception("Model loading failed.")
        raise RuntimeError(str(e))

    yield

    logger.info("Application shutdown.")

# ---------------- APP ----------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- HELPERS ----------------
def validate_image(upload_file: UploadFile):
    if not upload_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )


def save_upload(upload_file: UploadFile):
    unique_id = str(uuid.uuid4())

    filename = f"{unique_id}_input.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return unique_id, filename, file_path

# ---------------- ROUTES ----------------
@app.get("/")
def api_status():
    return {"status": "API is running"}


@app.post("/predict")
async def predict_and_generate_cams(
    file: UploadFile = File(...),
    mode: str = "resnet"
):
    validate_image(file)

    mode = mode.lower()

    if mode not in {"resnet", "fusion"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'resnet' or 'fusion'"
        )

    try:
        unique_id, input_filename, input_path = save_upload(file)

        if mode == "resnet":
            output_name = f"{unique_id}_resnet.jpg"
            output_path = os.path.join(RESULT_DIR, output_name)

            get_resnet_gradcam(
                input_path,
                resnet_predictor,
                output_path
            )

            selected_viz = f"/static/results/{output_name}"

            return {
                "status": "success",
                "mode": mode,
                "original_image": f"/static/uploads/{input_filename}",
                "selected_viz": selected_viz,
                "resnet_viz": selected_viz,
                "fusion_viz": None
            }

        output_name = f"{unique_id}_fusion.jpg"
        output_path = os.path.join(RESULT_DIR, output_name)

        get_fusion_gradcam(
            input_path,
            fusion_predictor,
            output_path
        )

        selected_viz = f"/static/results/{output_name}"

        return {
            "status": "success",
            "mode": mode,
            "original_image": f"/static/uploads/{input_filename}",
            "selected_viz": selected_viz,
            "resnet_viz": None,
            "fusion_viz": selected_viz
        }

    except Exception as e:
        logger.exception("GradCAM generation failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/resnet")
async def resnet_prediction(image: UploadFile = File(...)):
    validate_image(image)

    try:
        pil_image = Image.open(image.file).convert("RGB")

        result = resnet_predictor.resnet_predict(pil_image)

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as e:
        logger.exception("ResNet prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fusion")
async def fusion_prediction(image: UploadFile = File(...)):
    validate_image(image)

    try:
        pil_image = Image.open(image.file).convert("RGB")

        result = fusion_predictor.predict(pil_image)

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as e:
        logger.exception("Fusion prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/yolo")
async def yolo_detection(file: UploadFile = File(...)):
    validate_image(file)

    try:
        unique_id, input_filename, input_path = save_upload(file)

        output_name = f"{unique_id}_yolo.jpg"
        output_path = os.path.join(RESULT_DIR, output_name)

        result = get_yolo_damage_boxes(
            input_path,
            yolo_model,
            output_path
        )

        return {
            "status": "success",
            "original_image": f"/static/uploads/{input_filename}",
            "yolo_image": f"/static/results/{output_name}",
            "detections": result["detections"],
            "total_detections": result["total_detections"],
            "message": result["message"]
        }

    except Exception as e:
        logger.exception("YOLO detection failed.")
        raise HTTPException(status_code=500, detail=str(e))