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


# ---------------------------------------------------------------------------- #
#                               Data downloading                               #
# ---------------------------------------------------------------------------- #
df = pd.read_csv('data/EURUSDdays_ti.csv')
df.set_index('date', inplace=True)
# print(df.info())

# Train:Eval:Test = 70:20:10
window_size = 6
df_train = df.iloc[ 0 : int(df.shape[0]*0.7) ]
df_eval  = df.iloc[ int(df.shape[0]*0.7) - window_size: int(df.shape[0]*0.9) ]
df_test  = df.iloc[ int(df.shape[0]*0.9) - window_size: ]
# print(df_train.info())
# print(df_train.loc[:, 'close'])


# ---------------------------------------------------------------------------- #
#                               Environment test                               #
# ---------------------------------------------------------------------------- #
def test_env():
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


# ---------------------------------------------------------------------------- #
#                          Build Environment and Train                         #
# ---------------------------------------------------------------------------- #
def train_model():
    # Create environment
    def make_env(rank, seed=0):
        def _init():
            env = ForexEnv(df=df_train, window_size=window_size, trade_time=300)
            env.seed(seed + rank)
            return env
        set_random_seed(seed)
        return _init

    env_train = SubprocVecEnv([make_env(i) for i in range(4)])

    # env_train = ForexEnv(df=df_train, window_size=window_size, trade_time=300)
    # env_train = DummyVecEnv([lambda: env_train])

    # Custom Network Architecture: Custom actor (pi) and value function (vf) networks
    policy_kwargs = dict(activation_fn = th.nn.ReLU,
                         net_arch = [512, dict(pi=[256, 256], vf=[256, 256])])
    # Create the agent
    model = A2C(policy='MlpPolicy', 
                env = env_train, 
                policy_kwargs=policy_kwargs,
                tensorboard_log="./logs", 
                verbose=1) 
    
    # Train the agent
    model.learn(total_timesteps=10_000_000)

    print(" ------------------------- Finished !!! ------------------------- ")
    model.save("models/forex_rl")


# ---------------------------------------------------------------------------- #
#                                  Evaluation                                  #
# ---------------------------------------------------------------------------- #
def evaluate_model():
    env_eval = ForexEnv(df=df_eval, window_size=window_size)
    model = A2C.load("models/forex_rl")
    obs = env_eval.reset()
    while True:
        obs = obs[np.newaxis, ...]
        predict, _states = model.predict(obs)
        obs, rewards, done, info = env_eval.step(predict)
        if done:
            print("info", info)
            break

    env_eval.render_all()

def test_model():
    env_eval = ForexEnv(df=df_test, window_size=window_size)
    model = A2C.load("models/forex_rl")
    obs = env_eval.reset()
    while True:
        obs = obs[np.newaxis, ...]
        # predict = env.action_space.sample()
        predict, _states = model.predict(obs)
        obs, rewards, done, info = env_eval.step(predict)
        if done:
            print("info", info)
            break

    env_eval.render_all()

# normalized mean squared error (NMSE)
if __name__ == '__main__':
    train_model()
    # evaluate_model()
    # test_model()