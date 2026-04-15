import random


class Vehicle:
    def __init__(self):
        self.wait_time = 0
        self.crossed = False


class Lane:
    def __init__(self, name, config):
        self.name = name
        self.config = config

        self.vehicles = []
        self.max_capacity = config.get("lane_capacity", 10)
        self.spawn_prob = config.get("spawn_prob", 0.3)

    # =========================
    # RESET (FIXED)
    # =========================
    def reset(self):
        self.vehicles = []

    # =========================
    # SPAWN VEHICLES
    # =========================
    def spawn_vehicle(self):
        if len(self.vehicles) < self.max_capacity:
            if random.random() < self.spawn_prob:
                self.vehicles.append(Vehicle())

    # =========================
    # UPDATE LANE
    # =========================
    def update(self, is_green):
        self.spawn_vehicle()

        if is_green and len(self.vehicles) > 0:
            # First vehicle passes
            vehicle = self.vehicles.pop(0)
            vehicle.crossed = True

        # Increase wait time for remaining vehicles
        for v in self.vehicles:
            v.wait_time += 1

    # =========================
    # METRICS
    # =========================
    def get_queue_length(self):
        return len(self.vehicles)

    def get_avg_wait(self):
        if not self.vehicles:
            return 0
        return sum(v.wait_time for v in self.vehicles) / len(self.vehicles)

    def get_density(self):
        return len(self.vehicles) / self.max_capacity