from env.lane import Lane


class Intersection:
    def __init__(self, config):
        self.config = config

        # Create 4 lanes
        self.lanes = {
            "N": Lane("N", config),
            "S": Lane("S", config),
            "E": Lane("E", config),
            "W": Lane("W", config),
        }

        # Signal states: 0 = NS green, 1 = EW green
        self.signal = 0
        self.prev_signal = 0

        self.yellow_time = config.get("yellow_time", 2)
        self.min_green_time = config.get("min_green_time", 5)

        self.time_since_switch = 0
        self.in_yellow = False
        self.yellow_counter = 0

    # =========================
    # RESET
    # =========================
    def reset(self):
        for lane in self.lanes.values():
            lane.reset()

        self.signal = 0
        self.prev_signal = 0

        self.time_since_switch = 0
        self.in_yellow = False
        self.yellow_counter = 0

    # =========================
    # APPLY SIGNAL (SAFE LOGIC)
    # =========================
    def apply_signal(self, action):
        # If in yellow, ignore actions
        if self.in_yellow:
            return

        # Enforce minimum green time
        if action != self.signal and self.time_since_switch < self.min_green_time:
            return

        # Start yellow phase if switching
        if action != self.signal:
            self.in_yellow = True
            self.yellow_counter = 0
            self.prev_signal = self.signal
            self.signal = action
            self.time_since_switch = 0

    # =========================
    # UPDATE SIMULATION
    # =========================
    def update(self):
        # Handle yellow phase
        if self.in_yellow:
            self.yellow_counter += 1
            if self.yellow_counter >= self.yellow_time:
                self.in_yellow = False

            # During yellow, no lanes move
            for lane in self.lanes.values():
                lane.update(is_green=False)

            return

        # Determine which lanes are green
        if self.signal == 0:
            green_lanes = ["N", "S"]
        else:
            green_lanes = ["E", "W"]

        # Update lanes
        for name, lane in self.lanes.items():
            lane.update(is_green=(name in green_lanes))

        self.time_since_switch += 1

    # =========================
    # STATE VECTOR
    # =========================
    def get_state_vector(self):
        densities = [self.lanes[d].get_density() for d in ["N", "S", "E", "W"]]
        waits = [self.lanes[d].get_avg_wait() for d in ["N", "S", "E", "W"]]
        queues = [self.lanes[d].get_queue_length() for d in ["N", "S", "E", "W"]]

        return densities + waits + queues

    # =========================
    # REWARD (CALLED BY ENV)
    # =========================
    def compute_reward(self):
        # (Handled in traffic_env.py)
        return 0

    # =========================
    # SAFETY CHECK
    # =========================
    def is_transition_unsafe(self, action):
        # Prevent switching too fast or during yellow
        if self.in_yellow:
            return True

        if action != self.signal and self.time_since_switch < self.min_green_time:
            return True

        return False

    # =========================
    # RENDER (DEBUG VIEW)
    # =========================
    def render(self):
        print("\n=== INTERSECTION STATE ===")
        print("Signal:", "NS_GREEN" if self.signal == 0 else "EW_GREEN")
        print("Yellow Phase:", self.in_yellow)

        for name, lane in self.lanes.items():
            print(
                f"{name} | Queue: {lane.get_queue_length()} | "
                f"Avg Wait: {lane.get_avg_wait():.2f}"
            )