import numpy as np
import random


class RealTrafficEnv:
    def __init__(self, config):
        self.config = config

        self.timestep = 0
        self.max_steps = config["max_steps"]

        # Signal state
        self.signal = 0  # 0 = NS, 1 = EW
        self.timer = 0

        # Real-world like state (mock sensors)
        self.state = np.zeros(14)

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.timestep = 0
        self.timer = 0
        self.signal = 0

        self.state = self._get_sensor_data()

        return self.state

    # =========================
    # STEP
    # =========================
    def step(self, action):
        self.timestep += 1
        self.timer += 1

        # Apply action with safety constraints
        if self._is_safe(action):
            self.signal = action
            self.timer = 0

        # Fetch new "real" data
        self.state = self._get_sensor_data()

        reward = self._compute_reward()

        done = self.timestep >= self.max_steps

        return self.state, reward, done, {}

    # =========================
    # SENSOR INPUT (MOCK NOW)
    # =========================
    def _get_sensor_data(self):
        """
        Replace this later with:
        - Camera detection
        - IoT sensors
        - API feeds
        """

        densities = [random.randint(0, 20) for _ in range(4)]
        waits = [random.uniform(0, 30) for _ in range(4)]
        queues = [random.randint(0, 15) for _ in range(4)]

        return np.array(
            densities + waits + queues + [self.signal, self.timer],
            dtype=np.float32
        )

    # =========================
    # REWARD
    # =========================
    def _compute_reward(self):
        densities = self.state[:4]
        waits = self.state[4:8]

        reward = (
            -0.6 * sum(waits)
            -0.3 * sum(densities)
        )

        return reward

    # =========================
    # SAFETY
    # =========================
    def _is_safe(self, action):
        if self.timer < self.config["min_green_time"]:
            return False
        return True