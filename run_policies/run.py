import numpy as np
import time
from ..pushing_envs.pushing_env_lstm_categorical import PushingEnv
from sb3_contrib import RecurrentPPO

env = PushingEnv(graphics=True)
env.half_success_threshold()

model = RecurrentPPO.load("pushing-multimodal/run_policies/policies/ppo_lstm_categorical")

env.max_steps = 3000
env.disturbances = False
obs = env.reset()
lstm_states = None
num_envs = 1
episode_starts = np.ones((num_envs,), dtype=bool)
reward_ep = 0
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_starts = done
    reward_ep += reward
    time.sleep(1/30)
    if done:
        if reward == 50:
            print("Episode Successful")
        else:
            print("Episode Unsuccessful")
        reward_ep = 0
        obs = env.reset()
        lstm_states = None
        num_envs = 1
        episode_starts = np.ones((num_envs,), dtype=bool)
    