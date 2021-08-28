import argparse

from envs.forex_env import ForexEnv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    # ---------------------------------------------------------------------------- #
    #                               Environment tsdest                               #
    # ---------------------------------------------------------------------------- #

    df = pd.read_csv('data/EURUSDdays_ti.csv')
    df.set_index('date', inplace=True)
    # print(df.info())

    # Train:Eval:Test = 70:20:10
    window_size = 6
    df_test  = df.iloc[ int(df.shape[0]*0.9) - window_size: ]
    env_test = ForexEnv(df=df_test, window_size=window_size, trade_time=100)
    obs = env_test.reset()
    while True:
        obs = obs[np.newaxis, ...]
        predict = env_test.action_space.sample()
        obs, rewards, done, info = env_test.step(predict)
        if done:
            print("info", info)
            break

    env_test.render_all()


# normalized mean squared error (NMSE)
if __name__ == '__main__':
    main()
