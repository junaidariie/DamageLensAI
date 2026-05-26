import cv2
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def get_yolo_damage_boxes(image_path, yolo_model, output_path):
    logger.info("Starting YOLO damage detection...")

    try:
        image = Image.open(image_path).convert("RGB")

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
            logger.info(f"{len(boxes)} detections found.")

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
            logger.info("No detections found.")

        plotted = result.plot()

        cv2.imwrite(output_path, plotted)

        logger.info(f"YOLO output saved to: {output_path}")

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
        logger.exception("YOLO detection failed.")
        raise RuntimeError(f"YOLO failed: {str(e)}")