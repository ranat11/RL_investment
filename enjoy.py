import argparse

from envs.forex_env import ForexEnv
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnRewardThreshold,CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import torch as th

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def main():
    # ---------------------------------------------------------------------------- #
    #                               Data downloading                               #
    # ---------------------------------------------------------------------------- #

    window_size = 5
    env_eval = ForexEnv(env_data = 'data/EURUSDdays_ti.csv', env_type ="test", window_size=window_size)

    model = A2C.load("logs/models/a2c/ForexEnv-v0_1/ForexEnv-v0")
    obs = env_eval.reset()
    while True:
        obs = obs[np.newaxis, ...]
        # predict = env.action_space.sample()
        predict, _states = model.predict(obs)
        # print(predict)
        obs, rewards, done, info = env_eval.step(predict[0])
        if done:
            print("info", info)
            break

    env_eval.render()

if __name__ == '__main__':
    main()
