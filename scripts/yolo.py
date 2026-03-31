import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

yolo_model = YOLO("checkpoints/damage_detector.pt")


def get_yolo_damage_boxes(image_path, output_path):
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

        plotted_bgr = result.plot()
        plotted_rgb = plotted_bgr[..., ::-1]
        cv2.imwrite(output_path, plotted_rgb)

        return {
            "detections": detections,
            "total_detections": len(detections),
            "message": "No damage detected" if len(detections) == 0 else "Detections found"
        }

    except Exception as e:
        raise RuntimeError(f"YOLO failed: {str(e)}")