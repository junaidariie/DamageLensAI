import cv2
import logging
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def get_yolo_damage_boxes(image_path, yolo_model: YOLO, output_path: str):
    """Runs object detection using the YOLO ONNX model."""
    logger.info("Starting YOLO ONNX damage detection...")

    try:
        image = Image.open(image_path).convert("RGB")

        # Run ONNX inference through Ultralytics wrapper
        results = yolo_model.predict(
            source=image,
            conf=0.05,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes
        detections = []

        if boxes is not None and len(boxes) > 0:
            logger.info(f"{len(boxes)} damage detections found.")

            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                label = yolo_model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "box": [x1, y1, x2, y2]
                })
        else:
            logger.info("No damage detections found.")

        # Save bounding box visual overlay
        plotted = result.plot()
        cv2.imwrite(output_path, plotted)

        logger.info(f"YOLO ONNX output saved to: {output_path}")

        return {
            "detections": detections,
            "total_detections": len(detections),
            "message": (
                "No damage detected"
                if len(detections) == 0
                else "Detections found"
            )
        }

    except Exception as e:
        logger.exception("YOLO ONNX detection failed.")
        raise RuntimeError(f"YOLO detection failed: {str(e)}")