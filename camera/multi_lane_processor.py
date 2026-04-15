import cv2
from ultralytics import YOLO


class MultiLaneVehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

        # Vehicle classes (COCO)
        self.vehicle_classes = {2, 3, 5, 7}

        # Define lane regions (x1, y1, x2, y2)
        # ⚠️ You MUST tune these based on your camera
        self.lanes = {
            "lane_1": (0, 0, 320, 480),
            "lane_2": (320, 0, 640, 480),
            "lane_3": (640, 0, 960, 480),
            "lane_4": (960, 0, 1280, 480)
        }

    # =========================
    # DETECTION
    # =========================
    def detect(self, frame):
        results = self.model(frame)

        lane_counts = {lane: 0 for lane in self.lanes}
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])

                if cls_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Compute center of bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                lane_id = self._get_lane(cx, cy)

                if lane_id:
                    lane_counts[lane_id] += 1

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "lane": lane_id
                })

        return lane_counts, detections

    # =========================
    # LANE ASSIGNMENT
    # =========================
    def _get_lane(self, x, y):
        for lane, (x1, y1, x2, y2) in self.lanes.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return lane
        return None

    # =========================
    # DRAW
    # =========================
    def draw(self, frame, detections, lane_counts):
        # Draw lane regions
        for lane, (x1, y1, x2, y2) in self.lanes.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, lane, (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            lane = det["lane"]

            color = (0, 255, 0) if lane else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if lane:
                cv2.putText(frame, lane, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show counts
        y_offset = 30
        for lane, count in lane_counts.items():
            cv2.putText(frame, f"{lane}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

        return frame