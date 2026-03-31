import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from scripts.gradcam import get_resnet_gradcam, get_deit_gradcam
from scripts.yolo import get_yolo_damage_boxes
from scripts.prediction_helper import ResnetCarDamagePredictor, DeitCarDamagePredictor, FusionCarDamagePredictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

class_map = {
    0: "Front Breakage",
    1: "Front Crushed",
    2: "Front Normal",
    3: "Rear Breakage",
    4: "Rear Crushed",
    5: "Rear Normal"
}

resnet_checkpoint = "checkpoints/best_resnet_model.pt"
deit_checkpoint = "checkpoints/best_deit_model.pt"


Resnet_Model = ResnetCarDamagePredictor(resnet_checkpoint, class_map)
Deit_Model = DeitCarDamagePredictor(deit_checkpoint, class_map)
Fusion_Model = FusionCarDamagePredictor(resnet_predictor=Resnet_Model, deit_predictor=Deit_Model, resnet_weight=0.5, deit_weight=0.5)

resnet_predictor = Resnet_Model
deit_predictor = Deit_Model

# ====================== API Endpoint ======================

@app.get("/")
def api_status():
    return {"status": "API is running"}

# ============================= Grad-CAM Generation Endpoint =============================

@app.post("/predict")
async def predict_and_generate_cams(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}_input.jpg"
    resnet_out_name = f"{unique_id}_resnet.jpg"
    deit_out_name = f"{unique_id}_deit.jpg"

    input_path = os.path.join(UPLOAD_DIR, input_filename)
    resnet_path = os.path.join(RESULT_DIR, resnet_out_name)
    deit_path = os.path.join(RESULT_DIR, deit_out_name)

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate Grad-CAMs
    get_resnet_gradcam(input_path, resnet_predictor, resnet_path)
    get_deit_gradcam(input_path, deit_predictor, deit_path)

    # Return the URLs
    return {
        "status": "success",
        "original_image": f"/static/uploads/{input_filename}",
        "resnet_viz": f"/static/results/{resnet_out_name}",
        "deit_viz": f"/static/results/{deit_out_name}"
    }

# ============================= Prediction-Only Endpoints =============================
# ============================= Resnet Prediction =====================================

@app.post("/predict/resnet")
async def resnet_prediction(image : UploadFile = File(...)):
    try:
        image = Image.open(image.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    result = Resnet_Model.resnet_predict(image_input=image)
    return result

# ============================= Deit Prediction =====================================  
@app.post("/predict/deit")
async def deit_prediction(image : UploadFile = File(...)):
    try:
        image = Image.open(image.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    result = Deit_Model.deit_predict(image_input=image)
    return result

# ============================= Fusion Prediction ===================================== 
@app.post("/predict/fusion")
async def fusion_prediction(image : UploadFile = File(...)):
    try:
        image = Image.open(image.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    result = Fusion_Model.fuse_predict(image_input=image)
    return result

# ============================= YOLO Damage Box Endpoint =============================
@app.post("/predict/yolo")
async def yolo_detection(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())

    input_filename = f"{unique_id}_input.jpg"
    yolo_out_name = f"{unique_id}_yolo.jpg"

    input_path = os.path.join(UPLOAD_DIR, input_filename)
    yolo_path = os.path.join(RESULT_DIR, yolo_out_name)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = get_yolo_damage_boxes(input_path, yolo_path)

    return {
        "status": "success",
        "original_image": f"/static/uploads/{input_filename}",
        "yolo_image": f"/static/results/{yolo_out_name}",
        "detections": result["detections"],
        "total_detections": result["total_detections"],
        "message": result["message"]
    }