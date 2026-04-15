import numpy as np
import time
from env.gym_wrapper import TrafficGymEnv
from config import CONFIG


def choose_action(state):
    state = np.array(state)

    densities = state[0:4]
    waits = state[4:8]
    queues = state[8:12]

    score = (
        0.5 * densities +
        1.0 * waits +
        0.7 * queues
    )

    ns_score = score[0] + score[1]
    ew_score = score[2] + score[3]

    return 0 if ns_score >= ew_score else 1


def test():
    env = TrafficGymEnv(CONFIG)

    obs, _ = env.reset()

    for step in range(300):
        action = choose_action(obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # -------------------------
        # DEBUG PRINTS
        # -------------------------
        print(f"\nStep: {step}")
        print("Action:", "NS" if action == 0 else "EW")
        print("Reward:", round(reward, 2))

        # -------------------------
        # 🔥 RENDER WITH REWARD
        # -------------------------
        env.render(reward=reward)

        # Optional: slow down for better viewing
        time.sleep(0.05)

        if done:
            print("\n--- Episode Reset ---\n")
            obs, _ = env.reset()

    # Keep window open after simulation
    input("Press Enter to close visualization...")


if __name__ == "__main__":
    test()