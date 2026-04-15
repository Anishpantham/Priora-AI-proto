from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.gym_wrapper import TrafficGymEnv
from config import CONFIG


def train():
    # ✅ Pass config properly
    def make_env():
        return TrafficGymEnv(CONFIG)

    # Vectorized env (required for SB3 stability)
    env = make_vec_env(make_env, n_envs=1)

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99
    )

    # Train
    model.learn(total_timesteps=50000)

    # Save model
    model.save("ppo_traffic")

    print("✅ Training complete. Model saved as ppo_traffic.zip")


if __name__ == "__main__":
    train()