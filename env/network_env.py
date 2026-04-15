import numpy as np
from env.intersection import Intersection


class TrafficNetworkEnv:
    def __init__(self, config):
        self.config = config

        self.grid_size = config.get("grid_size", 2)  # 2x2 grid default

        # Create grid of intersections
        self.intersections = [
            [Intersection(config) for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]

        self.timestep = 0
        self.max_steps = config["max_steps"]

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.timestep = 0

        for row in self.intersections:
            for inter in row:
                inter.reset()

        return self._get_global_state()

    # =========================
    # STEP
    # =========================
    def step(self, actions):
        """
        actions: 2D list of actions for each intersection
        Example:
        [[0,1],
         [1,0]]
        """

        self.timestep += 1

        total_reward = 0

        # Apply actions
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                inter = self.intersections[i][j]
                action = actions[i][j]

                # Safety check
                if not inter.is_transition_unsafe(action):
                    inter.apply_signal(action)

        # Update all intersections
        for row in self.intersections:
            for inter in row:
                inter.update()
                total_reward += inter.compute_metrics()

        state = self._get_global_state()

        done = self.timestep >= self.max_steps

        return state, total_reward, done, {}

    # =========================
    # GLOBAL STATE
    # =========================
    def _get_global_state(self):
        """
        Flatten all intersection states into one vector
        """

        state = []

        for row in self.intersections:
            for inter in row:
                state.extend(inter.get_state_vector())

        return np.array(state, dtype=np.float32)

    # =========================
    # OPTIONAL: GLOBAL METRICS
    # =========================
    def compute_global_metrics(self):
        total_wait = 0
        total_density = 0

        for row in self.intersections:
            for inter in row:
                for lane in inter.lanes.values():
                    total_wait += lane.get_avg_wait()
                    total_density += lane.get_density()

        return {
            "total_wait": total_wait,
            "total_density": total_density
        }