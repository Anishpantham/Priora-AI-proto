import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.traffic_env import TrafficEnv


class TrafficGymEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        self.env = TrafficEnv(config)

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(12,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        terminated = done
        truncated = False

        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

    # ✅ FIXED RENDER
    def render(self, reward=0):
        return self.env.render(reward=reward)