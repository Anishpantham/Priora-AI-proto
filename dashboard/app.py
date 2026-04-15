import cv2
import numpy as np

from perception.multi_lane_detection import MultiLaneVehicleDetector
from env.real_env import RealTrafficEnv
from config import CONFIG

# If using Stable-Baselines3
from stable_baselines3 import PPO


class TrafficApp:
    def __init__(self):
        # =========================
        # LOAD COMPONENTS
        # =========================
        self.detector = MultiLaneVehicleDetector()
        self.env = RealTrafficEnv(CONFIG)

        # Load trained RL model
        self.model = PPO.load("traffic_model.zip")

        self.state = self.env.reset()

    # =========================
    # BUILD STATE FROM VISION
    # =========================
    def build_state(self, lane_counts):
        densities = [
            lane_counts["lane_1"],
            lane_counts["lane_2"],
            lane_counts["lane_3"],
            lane_counts["lane_4"]
        ]

        # Mock waits & queues (can be improved later)
        waits = [d * 2 for d in densities]
        queues = densities

        state = np.array(
            densities + waits + queues + [self.env.signal, self.env.timer],
            dtype=np.float32
        )

        return state

    # =========================
    # RUN LOOP
    # =========================
    def run(self):
        cap = cv2.VideoCapture(0)  # webcam / CCTV feed

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # =========================
            # DETECTION
            # =========================
            lane_counts, detections = self.detector.detect(frame)

            # =========================
            # BUILD STATE
            # =========================
            state = self.build_state(lane_counts)

            # =========================
            # RL DECISION
            # =========================
            action, _ = self.model.predict(state)

            # Step environment
            _, reward, done, _ = self.env.step(action)

            # =========================
            # DISPLAY SIGNAL
            # =========================
            signal_text = "NS GREEN" if action == 0 else "EW GREEN"

            # Draw detections
            frame = self.detector.draw(frame, detections, lane_counts)

            cv2.putText(frame, f"Signal: {signal_text}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(frame, f"Reward: {round(reward, 2)}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Smart Traffic System", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app = TrafficApp()
    app.run()