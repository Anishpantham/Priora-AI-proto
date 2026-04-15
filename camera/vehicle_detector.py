import cv2
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLO model
        self.model = YOLO(model_path)

        # Vehicle classes in COCO dataset
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    # =========================
    # DETECT VEHICLES
    # =========================
    def detect(self, frame):
        results = self.model(frame)

        detections = []
        vehicle_count = 0

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])

                if cls_id in self.vehicle_classes:
                    vehicle_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": self.vehicle_classes[cls_id]
                    })

        return vehicle_count, detections

    # =========================
    # DRAW DETECTIONS
    # =========================
    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame