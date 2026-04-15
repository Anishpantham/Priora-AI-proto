import random


class Vehicle:
    def __init__(self, direction):
        self.direction = direction

        # Motion properties
        self.speed = random.uniform(5, 10)
        self.max_speed = random.uniform(15, 25)
        self.acceleration = random.uniform(0.5, 1.5)

        # Position tracking
        self.position = 0

        # Waiting logic
        self.wait_time = 0

        # Status
        self.crossed = False

    # =========================
    # MOVE (GREEN SIGNAL)
    # =========================
    def move(self):
        """
        Accelerate smoothly and move forward
        """

        # Accelerate
        self.speed = min(self.speed + self.acceleration, self.max_speed)

        # Move forward
        self.position += self.speed

        # Check if crossed intersection
        if self.position > 100:
            self.crossed = True

        # Reduce waiting time gradually
        self.wait_time = max(0, self.wait_time - 1)

    # =========================
    # STOP (RED SIGNAL / QUEUE)
    # =========================
    def stop(self):
        """
        Decelerate smoothly instead of instant stop
        """

        # Gradual braking
        self.speed = max(0, self.speed - self.acceleration * 2)

        # Increase wait time if stopped
        if self.speed == 0:
            self.wait_time += 1

    # =========================
    # OPTIONAL (DEBUG)
    # =========================
    def __repr__(self):
        return f"Vehicle(pos={self.position:.2f}, speed={self.speed:.2f})"