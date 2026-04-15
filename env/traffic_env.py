import numpy as np
import matplotlib.pyplot as plt
from env.intersection import Intersection


class TrafficEnv:
    def __init__(self, config):
        self.config = config
        self.intersection = Intersection(config)

        self.timestep = 0
        self.max_steps = config["max_steps"]

        # =========================
        # VISUALIZATION INIT
        # =========================
        self.fig, self.ax = plt.subplots()
        plt.ion()  # interactive mode ON

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.timestep = 0
        self.intersection.reset()
        return self._get_state()

    # =========================
    # STEP
    # =========================
    def step(self, action):
        action = self._safe_action(action)

        # Apply signal
        self.intersection.apply_signal(action)

        # Update simulation
        self.intersection.update()

        # Get state
        state = self._get_state()

        # Reward
        reward = self._compute_reward()

        # Done condition
        self.timestep += 1
        done = self.timestep >= self.max_steps

        return state, reward, done, {}

    # =========================
    # STATE
    # =========================
    def _get_state(self):
        return np.array(self.intersection.get_state_vector(), dtype=np.float32)

    # =========================
    # REWARD
    # =========================
    def _compute_reward(self):
        lanes = self.intersection.lanes.values()

        total_wait = sum(l.get_avg_wait() for l in lanes)
        total_queue = sum(l.get_queue_length() for l in lanes)
        max_queue = max(l.get_queue_length() for l in lanes)

        throughput = sum(
            1 for lane in lanes
            for v in lane.vehicles if getattr(v, "crossed", False)
        )

        switch_penalty = 1 if self.intersection.signal != self.intersection.prev_signal else 0

        reward = (
            -1.2 * total_wait
            -1.0 * total_queue
            -2.0 * max_queue
            +0.8 * throughput
            -0.5 * switch_penalty
        )

        return reward

    # =========================
    # SAFE ACTION
    # =========================
    def _safe_action(self, action):
        if self.intersection.is_transition_unsafe(action):
            return self.intersection.signal
        return action

        # =========================
    # RENDER (🔥 UPGRADED)
    # =========================
    def render(self, reward=0):
        self.ax.clear()

        lanes = self.intersection.lanes

        # -------------------------
        # Draw Roads
        # -------------------------
        self.ax.plot([0, 10], [5, 5], 'black', linewidth=2)  # horizontal
        self.ax.plot([5, 5], [0, 10], 'black', linewidth=2)  # vertical

        # -------------------------
        # Draw Vehicles (🔥 SMART)
        # -------------------------
        spacing = 0.4

        for lane_id, lane in lanes.items():
            for i, vehicle in enumerate(lane.vehicles):

                # ✅ YOUR SNIPPET (used if positions exist)
                if hasattr(vehicle, "position_x") and hasattr(vehicle, "position_y"):
                    x = vehicle.position_x
                    y = vehicle.position_y

                # 🔁 Fallback (your old logic)
                else:
                    if lane_id == "N":
                        x, y = 5, 6 + i * spacing
                    elif lane_id == "S":
                        x, y = 5, 4 - i * spacing
                    elif lane_id == "E":
                        x, y = 6 + i * spacing, 5
                    elif lane_id == "W":
                        x, y = 4 - i * spacing, 5
                    else:
                        x, y = 5, 5  # safety

                # Color by direction
                color = "blue" if lane_id in ["N", "S"] else "red"

                self.ax.scatter(x, y, color=color, s=30)

        # -------------------------
        # Signal Display
        # -------------------------
        signal = self.intersection.signal

        self.ax.text(1, 9, f"Signal: {signal}", fontsize=12)
        self.ax.text(1, 8, f"Step: {self.timestep}", fontsize=12)
        self.ax.text(1, 7, f"Reward: {round(reward, 2)}", fontsize=12)

        # -------------------------
        # Limits & Style
        # -------------------------
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_title("RL Traffic Control")
        self.ax.axis('off')

        plt.pause(0.1)